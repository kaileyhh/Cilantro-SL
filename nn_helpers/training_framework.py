import pickle as pkl
import numpy as np

import os
from enum import Enum
from collections import Counter


class UQ(Enum):
    CONFORMAL = 1
    ENSEMBLE = 2
    MONDRIAN_CONFORMAL = 3


class Framework:
    def __init__(self, ve, pc, metrics_path=None):
        """
        ve: via_embedder.ViaEmbedder or via_film.ViaFilm object or None
        pc: pair_classifier.PairClassifier object or None
        """
        self.ve = ve
        self.pc = pc

        self.all_test = None
        self.all_train = None
        self.folds = None
        self.cv = None

        self.metrics_path = metrics_path
        self.metrics = {}

        with open(
            "/work/magroup/kaileyhu/datasets/SynLethSampled/all_pairs_NSP_EXP_5x.pkl",
            "rb",
        ) as f:
            pair_list = pkl.load(f)

        self.pair_list = pair_list

    def set_pc(self, pc):
        self.pc = pc

    def df_to_SL_embs(
        self,
        via_emb_path,
        SL_emb_path,
        batch_size=512,
        lr=0.001,
        num_epochs=100,
        name="tester",
        custom_func = None,
    ):
        if not self.ve:
            raise ValueError("ViaEmbedder object is not provided.")

        if not os.path.exists(via_emb_path):
            self.ve.init_regression(batch_size=batch_size, lr=lr, num_epochs=num_epochs)
            self.ve.train_and_test(
                0.0,
                f"/work/magroup/kaileyhu/synthetic_lethality/prediction/sCilantro/models/{name}.pth",
            )
            self.ve.setup_dataloaders(0.999)
            self.ve.test_nn()
    
            df2 = self.ve.df.drop(columns="viability score")
    
            self.ve.extract_nn_second_last(df2, via_emb_path)
        
        self.ve.create_SL_embs(self.pair_list, via_emb_path, SL_emb_path, custom_func=custom_func)

        print("Done with viability embedding extraction")

    def run_cv(
        self,
        all_test,
        all_train,
        folds=5,
        cv=1,
        batch_size=128,
        lr=0.0001,
        num_epochs=100,
        nn_save_paths = None,
        validation = False
    ):
        if not all_test:
            all_test, all_train = self.pc.setup_cv(folds, cv=cv)

        print(f"Running cross validation with {folds} folds, cv = {cv}")

        t_precision = 0
        t_recall = 0
        t_f1_score = 0
        t_auc = 0
        t_accuracy = 0
        t_aupr = 0

        prec_list = []
        recall_list = []
        f1_list = []
        auc_list = []
        acc_list = []
        aupr_list = []

        if nn_save_paths is None:
            nn_save_paths = [None for i in range(folds)]

        for i in range(folds):
            self.pc.init_classification(batch_size=batch_size, lr=lr, num_epochs=num_epochs)
            self.pc.setup_dataloaders_from_sets(all_test[i], all_train[i], validation=validation)
            self.pc.train_nn(nn_save_paths[i])
            vals, corr, _ = self.pc.test_nn()

            print(f"\n\nComputing accuracy results for fold {i} / {folds}")
            precision, recall, f1_score, auc, accuracy, aupr = self.pc.compute_acc(vals, corr)
            t_precision += precision / folds
            t_recall += recall / folds
            t_f1_score += f1_score / folds
            t_auc += auc / folds
            t_accuracy += accuracy / folds
            t_aupr += aupr / folds

            prec_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)
            auc_list.append(auc)
            acc_list.append(accuracy)
            aupr_list.append(aupr)
            print(f"Appended results to list")
            print("\n\n")

        print(
            f"accuracy,precision,recall,f1 score, auc, aupr\n{t_accuracy}\n{t_precision}\n{t_recall}\n{t_f1_score}\n{t_auc}\n{t_aupr}\n"
        )
        print("\n\n")

        self.metrics["general_results"] = {
            "accuracy": t_accuracy,
            "precision": t_precision,
            "recall": t_recall,
            "f1": t_f1_score,
            "auc": t_auc,
            "aupr": t_aupr,
        }

        self.metrics["individual_results"] = {
            "accuracy": acc_list,
            "precision": prec_list,
            "recall": recall_list,
            "f1": f1_list,
            "auc": auc_list,
            "aupr": aupr_list,
        }

        if self.metrics_path:
            with open(self.metrics_path, "wb") as f:
                pkl.dump(self.metrics, f)

        self.all_test = all_test
        self.all_train = all_train
        self.folds = folds
        self.cv = cv

    def uncertainty_quantification(
        self,
        uq_type,
        net=None,
        num_ensemble=5,
        batch_size=128,
        lr=0.0001,
        num_epochs=100,
        mondrian_class_dict={},
        extract_name=lambda x: x,
    ):
        if "uncertainty_quantification" not in self.metrics:
            self.metrics["uncertainty_quantification"] = []

        if uq_type == UQ.MONDRIAN_CONFORMAL:  # A new UQ type
            print(f"Running Mondrian conformal prediction with {self.folds} folds")

            if net is None:
                net = [None for i in range(self.folds)]

            t_pred_vals = []
            t_corr = []
            t_conf_scores = []
            t_test_names = []
            t_probs_list = []
            t_calib = []

            for i in range(self.folds):
                self.pc.load_net(net[i])
                calib_dl, test_dl = self.pc.get_calib_test(self.all_test[i], 0.1, mondrian_class_dict=mondrian_class_dict, extract_name=extract_name)
                # calib_names = list(self.pc.calib_names['patient'])
                # test_names = list(self.pc.test_names['patient'])
                # calib_class_dict = [mondrian_class_dict[k] for k in calib_names]
                # test_class_dict = [mondrian_class_dict[k] for k in test_names]

                # c_dict = dict(Counter(calib_class_dict))
                # t_dict = dict(Counter(test_class_dict))

                # perc_dict = {}
                # for k in c_dict:
                #     perc_dict[k] = c_dict[k] / (c_dict[k] + t_dict[k])
                # print(f"Distribution of calibration is {perc_dict}")

                vals, corr, conf_scores, test_names, calibration_scores, probs_list = self.pc.conformal_pred(
                    calib_dl, test_dl, mondrian_class_dict=mondrian_class_dict, extract_name=extract_name
                )
                t_pred_vals.append(vals)
                t_corr.append(corr)
                t_conf_scores.append(conf_scores)
                t_test_names.append(test_names)
                t_calib.append(calibration_scores)
                t_probs_list.append(probs_list)

            self.metrics["mondrian_pred_vals"] = t_pred_vals
            self.metrics["mondrian_corr"] = t_corr
            self.metrics["mondrian_conf_scores"] = t_conf_scores
            self.metrics["mondrian_test_names"] = t_test_names
            self.metrics["mondrian_probs_list"] = t_probs_list
            self.metrics["calibration_scores"] = t_calib

            self.metrics["uncertainty_quantification"].append(uq_type)
            if self.metrics_path:
                with open(self.metrics_path, "wb") as f:
                    pkl.dump(self.metrics, f)

        elif uq_type == UQ.CONFORMAL:
            print(f"Running conformal prediction with {self.folds} folds")
            self.pc.load_net(net)

            t_pred_vals = []
            t_corr = []
            t_conf_scores = []
            t_test_names = []

            for i in range(self.folds):
                calib_dl, test_dl = self.pc.get_calib_test(self.all_test[i], 0.5)
                vals, corr, conf_scores, test_names, alpha_to_quantile = self.pc.conformal_pred(
                    calib_dl, test_dl
                )
                t_pred_vals.append(vals)
                t_corr.append(corr)
                t_conf_scores.append(conf_scores)
                t_test_names.append(test_names)

            self.metrics["conformal_pred_vals"] = t_pred_vals
            self.metrics["conformal_corr"] = t_corr
            self.metrics["conformal_conf_scores"] = t_conf_scores
            self.metrics["conformal_test_names"] = t_test_names
            self.metrics["conformal_alpha_to_quantile"] = alpha_to_quantile

            self.metrics["uncertainty_quantification"].append(uq_type)
            if self.metrics_path:
                with open(self.metrics_path, "wb") as f:
                    pkl.dump(self.metrics, f)

        elif uq_type == UQ.ENSEMBLE:
            print(f"Running ensemble with {num_ensemble} models per fold, {self.folds} folds")
            t_pred_vals = []
            t_variance = []
            t_corr = []
            t_test_names = []

            for i in range(self.folds):
                fold_preds = []
                for j in range(num_ensemble):
                    self.pc.init_classification(batch_size=batch_size, lr=lr, num_epochs=num_epochs)
                    self.pc.setup_dataloaders_from_sets(self.all_test[i], self.all_train[i])
                    self.pc.train_nn(None)
                    vals, corr, _ = self.pc.test_nn()
                    fold_preds.append(vals)

                t_pred_vals.append(fold_preds)
                t_variance.append(np.var(fold_preds, axis=0))
                t_corr.append(corr)
                t_test_names.append(self.pc.test_names)

            self.metrics["ensemble_pred_vals"] = t_pred_vals
            self.metrics["ensemble_variance"] = t_variance
            self.metrics["ensemble_corr"] = t_corr
            self.metrics["ensemble_test_names"] = t_test_names

            self.metrics["uncertainty_quantification"].append(uq_type)

            if self.metrics_path:
                with open(self.metrics_path, "wb") as f:
                    pkl.dump(self.metrics, f)
        else:
            raise ValueError("Unknown uncertainty quantification type.")
