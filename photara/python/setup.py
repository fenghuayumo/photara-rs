from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    """Mark the wheel as platform-specific because it ships a native .pyd."""

    def has_ext_modules(self) -> bool:
        return True


setup(distclass=BinaryDistribution)
