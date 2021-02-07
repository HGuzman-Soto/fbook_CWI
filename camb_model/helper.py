from sklearn.pipeline import FeatureUnion, Pipeline


def extract_feature_names(model, name):
    """Extracts the feature names from arbitrary sklearn models

    Args:
      model: The Sklearn model, transformer, clustering algorithm, etc. which we want to get named features for.
      name: The name of the current step in the pipeline we are at.

    Returns:
      The list of feature names.  If the model does not have named features it constructs feature names
      by appending an index to the provided name.
    """
    if hasattr(model, "get_feature_names"):
        return model.get_feature_names()
    elif hasattr(model, "n_clusters"):
        return [f"{name}_{x}" for x in range(model.n_clusters)]
    elif hasattr(model, "n_components"):
        return [f"{name}_{x}" for x in range(model.n_components)]
    elif hasattr(model, "components_"):
        n_components = model.components_.shape[0]
        return [f"{name}_{x}" for x in range(n_components)]
    elif hasattr(model, "classes_"):
        return classes_
    else:
        return [name]


def get_feature_names(model, names):
    """Thie method extracts the feature names in order from a Sklearn Pipeline

    This method only works with composed Pipelines and FeatureUnions.  It will
    pull out all names using DFS from a model.

    Args:
        model: The model we are interested in
        names: The list of names of final featurizaiton steps
        name: The current name of the step we want to evaluate.

    Returns:
        feature_names: The list of feature names extracted from the pipeline.
    """

    # Check if the name is one of our feature steps.  This is the base case.
    name = ""
    if name in names:
        # If it has the named_steps atribute it's a pipeline and we need to access the features
        if hasattr(model, "named_steps"):
            return extract_feature_names(model.named_steps[name], name)
        # Otherwise get the feature directly
        else:
            return extract_feature_names(model, name)
    elif type(model) is Pipeline:
        feature_names = []
        for name in model.named_steps.keys():
            feature_names += get_feature_names(
                model.named_steps[name], names, name)
        return feature_names
    elif type(model) is FeatureUnion:
        feature_names = []
        for name, new_model in model.transformer_list:
            feature_names += get_feature_names(new_model, names, name)
        return feature_names
    # If it is none of the above do not add it.
    else:
        return []
