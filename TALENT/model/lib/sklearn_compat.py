"""Estimator validation across supported scikit-learn versions."""

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # scikit-learn 1.4 and 1.5
    def validate_data(estimator, *args, **kwargs):
        if "skip_check_array" in kwargs:
            kwargs["cast_to_ndarray"] = not kwargs.pop("skip_check_array")
        if "ensure_all_finite" in kwargs:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
        return estimator._validate_data(*args, **kwargs)
