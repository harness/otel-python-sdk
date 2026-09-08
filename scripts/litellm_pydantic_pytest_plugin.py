"""Pytest plugin to rebuild litellm pydantic models in the test process."""


def pytest_configure(config):  # pylint: disable=unused-argument
    try:
        from litellm.types.llms.openai import (  # noqa: F401
            ChatCompletionReasoningSummaryTextBlock,
        )
        import litellm.types.utils as litellm_utils
        from pydantic import BaseModel
    except ImportError:
        return

    for name in dir(litellm_utils):
        obj = getattr(litellm_utils, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and hasattr(obj, "model_rebuild")
        ):
            try:
                obj.model_rebuild()
            except Exception:
                pass
