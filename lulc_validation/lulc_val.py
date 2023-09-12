import pandas as pd
import math


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
        self.samples_df["oa_indicator"] = (
            self.samples_df[self.ref_class] == self.samples_df[self.map_class]
        ) * 1

    def _correct_classification_indicator(self):
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
        """Compute user's accuracy accounting for reference data generated via stratified sampling."""
        n_df = self._n_samples_per_strata()

        users_accuracy = {}

        for c in self.class_list:
            numerator = 0
            denominator = 0

            for i, h in enumerate(self.strata_list):
                # total N for the strata
                N_h = self.n_strata[i]

                # sampled n from N in strata
                n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]

                # df for the strata
                y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :]

                # equation 27 of Stehman (2014)
                y_hat_num = y_df.loc[
                    (y_df["oa_indicator"] == 1) & (y_df[self.map_class] == c), :
                ].shape[0]
                if y_hat_num == 0:
                    y_hat_num = 0
                else:
                    y_hat_num = y_hat_num / n_h_samples

                x_hat_denom = y_df.loc[(y_df[self.map_class] == c), :].shape[0]
                if x_hat_denom == 0:
                    x_hat_denom = 0
                else:
                    x_hat_denom = x_hat_denom / n_h_samples

                numerator += N_h * y_hat_num
                denominator += N_h * x_hat_denom

            if denominator == 0:
                users_accuracy[str(c)] = 0
            else:
                users_accuracy[str(c)] = numerator / denominator

        return users_accuracy

    def producers_accuracy(self):
        """Compute producer's accuracy account for reference data generated via stratified sampling."""
        n_df = self._n_samples_per_strata()

        producers_accuracy = {}

        for c in self.class_list:
            numerator = 0
            denominator = 0

            for i, h in enumerate(self.strata_list):
                # total N for the strata
                N_h = self.n_strata[i]

                # sampled n from N in strata
                n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]

                # df for the strata
                y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :]

                # equation 27 of Stehman (2014)
                y_hat_num = y_df.loc[
                    (y_df["oa_indicator"] == 1) & (y_df[self.ref_class] == c), :
                ].shape[0]
                if y_hat_num == 0:
                    y_hat_num = 0
                else:
                    y_hat_num = y_hat_num / n_h_samples

                x_hat_denom = y_df.loc[(y_df[self.ref_class] == c), :].shape[0]
                if x_hat_denom == 0:
                    x_hat_denom = 0
                else:
                    x_hat_denom = x_hat_denom / n_h_samples

                numerator += N_h * y_hat_num
                denominator += N_h * x_hat_denom

            if denominator == 0:
                producers_accuracy[str(c)] = 0
            else:
                producers_accuracy[str(c)] = numerator / denominator

        return producers_accuracy

    def accuracy(self):
        """Compute overall accuracy accounting for reference data generated via stratified sampling."""
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

    def accuracy_se(self):
        """Compute standard errors for overall accuracy."""
        c_df = self._correct_classification_indicator()
        n_df = self._n_samples_per_strata()
        u_df = pd.merge(c_df, n_df, on=self.strata_col, how="inner")

        N = sum(self.n_strata)
        N_sq = N * N

        h_tmp = 0
        for i, h in enumerate(self.strata_list):
            # total N for the strata
            N_h = self.n_strata[i]

            N_h_sq = N_h * N_h

            # sampled n from N in strata
            n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]

            y_u = u_df.loc[u_df[self.strata_col] == h, :]
            y_u = y_u["oa_indicator"] / y_u[self.strata_col + "_n"]
            y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :]

            # equation 26 of Stehman (2014)
            y_u_y_h_diff_sq = (y_df["oa_indicator"] - y_u.iloc[0]) * (y_df["oa_indicator"] - y_u.iloc[0])
            numerator_tmp = sum(y_u_y_h_diff_sq)
            denominator_tmp = (n_h_samples - 1)
            s_2_yh = numerator_tmp / denominator_tmp 

            # SE for overall accuracy worked example
            numerator = (1 - n_h_samples / N_h) * s_2_yh
            denominator = n_h_samples
            tmp_h = N_h_sq * (numerator / denominator)
            h_tmp += tmp_h
        
        SE_y = (1/N_sq) * h_tmp

        return math.sqrt(SE_y)
    
    def users_accuracy_se(self):
        """Compute standard errors for User's accuracy"""
        n_df = self._n_samples_per_strata()

        N = sum(self.n_strata)

        users_acc = self.users_accuracy()
        users_accuracy_se = {}

        for c in self.class_list:
            R = users_acc[str(c)]
            h_tmp = 0
            x_tmp = 0
            for i, h in enumerate(self.strata_list):
                N_h = self.n_strata[i]

                N_h_sq = N_h * N_h

                n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]

                y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :].copy()

                # X indicator function for user's accuracy
                y_df.loc[:, "x_indicator"] = (y_df[self.map_class] == c) * 1
                y_df.loc[:, "y_u_indicator"] = (y_df["oa_indicator"] == 1) & (y_df[self.map_class] == c) * 1

                # s^2_{yh} - equation 26
                y_hat_h = y_df.loc[
                    (y_df["oa_indicator"] == 1) & (y_df[self.map_class] == c), :
                ].shape[0]
                if y_hat_h == 0:
                    y_hat_h = 0
                else:
                    y_hat_h = y_hat_h / n_h_samples
                
                y_u_y_h_diff_sq = (y_df["y_u_indicator"] - y_hat_h) * (y_df["y_u_indicator"] - y_hat_h) /  (n_h_samples - 1)
                s_2_yh = sum(y_u_y_h_diff_sq)

                # s^2_{xh} - equation 26
                x_hat_h = y_df.loc[(y_df[self.map_class] == c), :].shape[0]
                if x_hat_h == 0:
                    x_hat_h = 0
                else:
                    x_hat_h = x_hat_h / n_h_samples

                x_u_x_h_diff_sq = (y_df["x_indicator"] - x_hat_h) * (y_df["x_indicator"] - x_hat_h) / (n_h_samples - 1)
                s_2_xh = sum(x_u_x_h_diff_sq)
                
                # s_{xyh} - equation 29
                s_xyh = sum(((y_df["y_u_indicator"] - y_hat_h) * (y_df["x_indicator"] - x_hat_h)) / (n_h_samples - 1))

                # equation 28 of Stehman (2014)
                tmp_h = N_h_sq * (1 - n_h_samples / N_h) * (s_2_yh + ((R * R) * s_2_xh) - (2 * R * s_xyh))
                h_tmp += (tmp_h / n_h_samples)
                
                x_tmp += N_h * x_hat_h
            
            X_sq = x_tmp * x_tmp

            users_accuracy_se[str(c)] = math.sqrt((1 / X_sq) * h_tmp)
        
        return users_accuracy_se

    def producers_accuracy_se(self):
        """Compute standard errors for Producer's accuracy"""
        n_df = self._n_samples_per_strata()

        N = sum(self.n_strata)

        producers_acc = self.producers_accuracy()
        producers_accuracy_se = {}

        for c in self.class_list:
            R = producers_acc[str(c)]
            h_tmp = 0
            x_tmp = 0
            for i, h in enumerate(self.strata_list):
                N_h = self.n_strata[i]

                N_h_sq = N_h * N_h

                n_h_samples = n_df.loc[n_df[self.strata_col] == h, :].iloc[0, 1]

                y_df = self.samples_df.loc[self.samples_df[self.strata_col] == h, :].copy()

                # X indicator function for user's accuracy
                y_df.loc[:, "x_indicator"] = (y_df[self.ref_class] == c) * 1
                y_df.loc[:, "y_u_indicator"] = (y_df["oa_indicator"] == 1) & (y_df[self.ref_class] == c) * 1

                # s^2_{yh} - equation 26
                y_hat_h = y_df.loc[
                    (y_df["oa_indicator"] == 1) & (y_df[self.ref_class] == c), :
                ].shape[0]
                if y_hat_h == 0:
                    y_hat_h = 0
                else:
                    y_hat_h = y_hat_h / n_h_samples
                
                y_u_y_h_diff_sq = (y_df["y_u_indicator"] - y_hat_h) * (y_df["y_u_indicator"] - y_hat_h) /  (n_h_samples - 1)
                s_2_yh = sum(y_u_y_h_diff_sq)

                # s^2_{xh} - equation 26
                x_hat_h = y_df.loc[(y_df[self.ref_class] == c), :].shape[0]

                if x_hat_h == 0:
                    x_hat_h = 0
                else:
                    x_hat_h = x_hat_h / n_h_samples

                x_u_x_h_diff_sq = (y_df["x_indicator"] - x_hat_h) * (y_df["x_indicator"] - x_hat_h) / (n_h_samples - 1)
                s_2_xh = sum(x_u_x_h_diff_sq)   
                
                # s_{xyh} - equation 29
                s_xyh = sum(((y_df["y_u_indicator"] - y_hat_h) * (y_df["x_indicator"] - x_hat_h)) / (n_h_samples - 1))

                # equation 28 of Stehman (2014)
                tmp_h = N_h_sq * (1 - n_h_samples / N_h) * (s_2_yh + ((R * R) * s_2_xh) - (2 * R * s_xyh))
                h_tmp += (tmp_h / n_h_samples)
                
                x_tmp += N_h * x_hat_h
            
            X_sq = x_tmp * x_tmp

            producers_accuracy_se[str(c)] = math.sqrt((1 / X_sq) * h_tmp)
        
        return producers_accuracy_se



            

