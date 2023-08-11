import pandas as pd


class StratVal:
    def __init__(
        self,
        strata_list: list,
        class_list: list,
        n_strata: list,
        samples_df: pd.DataFrame,
        strata_col: str,
        ref_class: str,
        map_class: str,
    ):
        """LULC accuracy assessment with stratified samples.

        Parameters
        ----------
        strata_list : list
            List of labels for strata.
        class_list : list
            List of labels for LULC map classes.
        n_strata : list
            List of the total number of pixels in each strata.
        samples_df : pd.DataFrame
            pandas DataFrame
        strata_col : str
            Column label for strata in `samples_df`
        ref_class : str
            Column label for reference classes in `samples_df`
        map_class : str
            Column label for map classes in `samples_df`
        """
        self.strata_list = strata_list
        self.class_list = class_list
        self.n_strata = n_strata
        self.samples_df = samples_df
        self.strata_col = strata_col
        self.ref_class = ref_class
        self.map_class = map_class

    def _correct_classification_indicator(self):
        self.samples_df["oa_indicator"] = (
            self.samples_df[self.ref_class] == self.samples_df[self.map_class]
        ) * 1
        correct_classified_by_strata = (
            self.samples_df.groupby([self.strata_col])[["oa_indicator"]]
            .sum()
            .reset_index()
        )

        return correct_classified_by_strata

    def _n_samples_per_strata(self):
        samples_strata_count = (
            self.samples_df.groupby([self.strata_col]).count().reset_index()
        )
        samples_strata_count = samples_strata_count.loc[
            :, [self.strata_col, self.map_class]
        ]
        samples_strata_count.columns = [self.strata_col, self.strata_col + "_n"]
        return samples_strata_count

    def users_accuracy(self):
        """Compute user's accuracy account for reference data generated via stratified sampling."""
        n_df = self._n_samples_per_strata()
        self.samples_df["oa_indicator"] = (
            self.samples_df[self.ref_class] == self.samples_df[self.map_class]
        ) * 1

        users_accuracy = {}

        for c in self.class_list:
            numerator = 0
            denominator = 0

            for i, h in enumerate(self.strata_list):
                N_h = self.n_strata[i]
                n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]
                y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :]

                y_h_num = y_df.loc[
                    (y_df["oa_indicator"] == 1) & (y_df[self.map_class] == c), :
                ].shape[0]
                if y_h_num == 0:
                    y_h_num = 0
                else:
                    y_h_num = y_h_num / n_h_samples

                x_h_denom = y_df.loc[(y_df[self.map_class] == c), :].shape[0]
                if x_h_denom == 0:
                    x_h_denom = 0
                else:
                    x_h_denom = x_h_denom / n_h_samples

                numerator += N_h * y_h_num
                denominator += N_h * x_h_denom

            if denominator == 0:
                users_accuracy[str(c)] = 0
            else:
                users_accuracy[str(c)] = numerator / denominator

        return users_accuracy

    def producers_accuracy(self):
        """Compute producer's accuracy account for reference data generated via stratified sampling."""
        n_df = self._n_samples_per_strata()
        self.samples_df["oa_indicator"] = (
            self.samples_df[self.ref_class] == self.samples_df[self.map_class]
        ) * 1

        producers_accuracy = {}

        for c in self.class_list:
            numerator = 0
            denominator = 0

            for i, h in enumerate(self.strata_list):
                N_h = self.n_strata[i]
                n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]
                y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :]

                y_h_num = y_df.loc[
                    (y_df["oa_indicator"] == 1) & (y_df[self.map_class] == c), :
                ].shape[0]
                if y_h_num == 0:
                    y_h_num = 0
                else:
                    y_h_num = y_h_num / n_h_samples

                x_h_denom = y_df.loc[(y_df[self.ref_class] == c), :].shape[0]
                if x_h_denom == 0:
                    x_h_denom = 0
                else:
                    x_h_denom = x_h_denom / n_h_samples

                numerator += N_h * y_h_num
                denominator += N_h * x_h_denom

            if denominator == 0:
                producers_accuracy[str(c)] = 0
            else:
                producers_accuracy[str(c)] = numerator / denominator

        return producers_accuracy

    def accuracy(self):
        """Compute overall accuracy account for reference data generated via stratified sampling."""
        numerator = 0
        denominator = 0
        c_df = self._correct_classification_indicator()
        n_df = self._n_samples_per_strata()
        u_df = pd.merge(c_df, n_df, on=self.strata_col, how="inner")

        N = sum(self.n_strata)
        for i, h in enumerate(self.strata_list):
            N_h = self.n_strata[i]
            y_u = u_df.loc[u_df[self.strata_col] == h, :]
            y_u = y_u["oa_indicator"] / y_u[self.strata_col + "_n"]
            N_h_y_u = N_h * y_u.iloc[0]
            numerator += N_h_y_u
            denominator += N_h

        return numerator / denominator
