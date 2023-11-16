from metaflow import (
    FlowSpec,
    Parameter,
    batch,
    environment,
    pip_base,
    project,
    step,
)

PIP_ENV = {
    "ml_verifications[all]": "3.1.0",
}

PYTHON = "3.10.6"


class Loan:
    amount: int
    default_prob: float

    def __init__(self, amount: int, default_prob: float):
        self.amount = amount
        self.default_prob = default_prob

    def __repr__(self):
        return f"Loan(amount={self.amount}, default_prob={self.default_prob})"

    def get_exp_loss(self):
        return self.amount * self.default_prob

@pip_base(packages=PIP_ENV, python=PYTHON)
class HackFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd

        training_data_path = "data/train_data.parquet"
        self.training_df = pd.read_parquet(training_data_path)
        print("loaded training data")

        n_estimators = list(range(300, 900, 50)) + [587]
        max_depth = list(range(10, 20))
        min_child_weight = list(range(10, 40))
        gamma = list(range(4,10))

        self.tups = zip(
            n_estimators,
            max_depth,
            min_child_weight,
            gamma,
        )

        self.next(self.fit, foreach="tups")

    @batch(cpu=8, memory=10000)
    @step
    def fit(self):
        from sklearn.model_selection import train_test_split

        from verifications.models.transformers.pipeline_recipes import (
            PipelineWithSampleIndependentPandasTransformersRecipe,
        )
        from verifications.models.transformers.defaults import DefaultXgbClassifier

        self.n_estimator = self.input[0]
        self.max_depth = self.input[1]
        self.min_child_weight = self.input[2]
        self.gamma = self.input[3]

        print(f"n_estimator: {self.n_estimator}")

        dtypes = self.training_df.drop(columns=["upstart_id", "default"]).dtypes.reset_index()
        dtypes.columns = ["feature", "dtype"]
        self.columns_categorical = list(
            dtypes[(dtypes["dtype"] == "bool") | (dtypes["dtype"] == "object")]["feature"].values
        )
        self.columns_numeric = list(set(dtypes.feature) - set(self.columns_categorical))

        estimator = DefaultXgbClassifier
        params = dict(
            colsample_bytree=0.8947961236633096,
            gamma=self.gamma,
            learning_rate=0.03608530141207742,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            n_estimators=self.n_estimator,
            reg_alpha=0,  # Not tuned
            reg_lambda=1,  # Not tuned
            subsample=0.829162364031506,
        )
        estimator.set_params(**params)
        print("set params")


        pipeline_recipe = PipelineWithSampleIndependentPandasTransformersRecipe(
            columns_input=self.columns_categorical + self.columns_numeric,
            columns_numeric=self.columns_numeric,
            columns_categorical=self.columns_categorical,
            sample_independent_transformers=[],
            estimator=estimator,
        )
        self.pipeline = pipeline_recipe()
        print("pipeline created")


        X = self.training_df[self.columns_categorical + self.columns_numeric]
        y = self.training_df["default"]
        print("got X and y")


        test_size = 0.2
        random_state = 42
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )
        print("split train test data")


        self.pipeline.fit(X_train, y_train)
        print("fitting complete")

        self.next(self.eval)


    @step
    def eval(self):
        from sklearn.metrics import roc_auc_score

        preds = self.pipeline.predict_proba(self.X_test)[:, 1]
        self.metric = roc_auc_score(self.y_test, preds)

        self.next(self.join)


    @step
    def join(self, inputs):
        self.final_pl = None
        max_metric = 0
        for obj in inputs:
            print(obj.n_estimator, obj.metric)
            if max_metric < obj.metric:
                max_metric = obj.metric
                self.final = obj.pipeline
        self.next(self.end)


    @step
    def end(self):
        pass


if __name__ == "__main__":
    HackFlow()
