def require_fitted(f):
    """
    Use as a decorator on all methods that require a fitted instance (e.g. the
    transform method in preprocessors).
    """

    def _f(self, *args, **kwargs):
        if not self.is_fitted:
            raise RuntimeError(
                f"Method {f.__name__} cannot be called before calling .fit()!"
            )
        return f(self, *args, **kwargs)

    return _f
