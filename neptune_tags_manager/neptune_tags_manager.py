import argparse
import neptune


class NeptuneTagsManager:
    # The different tags.
    __OPTUNA_TAG = "Optuna"
    __MULTI_OBJECTIVE_TAG = "Multi-Objective"
    __FULL_TRAIN_TAG = "full_train"
    __PRIOR_SAMPLING_TAG = "prior_sampling"
    __SKD_SAMPLING_TAG = "SKD_sampling"

    # The path to the tags in a neptune run.
    __RUN_TAGS_PATH = "sys/tags"

    @staticmethod
    def check_attribute(obj, attr):
        return hasattr(obj, attr) and getattr(obj, attr)

    @classmethod
    def add_tags(cls, run: neptune.Run, args: argparse.Namespace, is_optuna: bool = False):
        tags = ["without_post"]

        # Add the optuna tag if needed.
        if is_optuna:
            tags.append(cls.__OPTUNA_TAG)

            # Add the multi-objective only if it exists and true.
            if cls.check_attribute(args, "multi_objective"):
                tags.append(cls.__MULTI_OBJECTIVE_TAG)

        # Otherwise add the full train tag.
        else:
            tags.append(cls.__FULL_TRAIN_TAG)

        # Add the sampling type.
        if cls.check_attribute(args, "prior_sampling"):
            tags.append(cls.__PRIOR_SAMPLING_TAG)

        else:
            tags.append(cls.__SKD_SAMPLING_TAG)

        run[cls.__RUN_TAGS_PATH].add(tags)





