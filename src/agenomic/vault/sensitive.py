"""Write-only wrapper for values that may be sent to the vault and never read back.

A secret value enters the SDK through :class:`Sensitive` and leaves it only
inside the request body of the call that stores it. Every rendering of the
wrapper (``repr``, ``str``, ``format``, pickle, copy, JSON and pydantic
serialization) is the same constant mask whatever the length of the value,
and the class has no public accessor: the transport is the only reader.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, SecretStr
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

MASK = "**********"


class SensitiveUnavailable(ValueError):
    """The wrapped value is not available: it was masked, consumed or never set.

    Example:
        >>> import pickle
        >>> restored = pickle.loads(pickle.dumps(Sensitive("correct horse battery")))
        >>> str(restored)
        '**********'
    """


class Sensitive:
    """A string that can be passed in but not read back.

    Example:
        >>> value = Sensitive("correct horse battery staple")
        >>> repr(value), str(value), f"{value:>40}"
        ("Sensitive('**********')", '**********', '**********')
        >>> import json
        >>> json.dumps({"value": value}, default=str)
        '{"value": "**********"}'
    """

    __slots__ = ("_held",)
    _held: Optional[str]

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Sensitive wraps a str, not {type(value).__name__}")
        object.__setattr__(self, "_held", value)

    @classmethod
    def from_env(cls, name: str) -> Sensitive:
        """Wrap the value of an environment variable; the value is never echoed.

        Example:
            >>> Sensitive.from_env("AGENOMIC_DOCTEST_UNSET_VARIABLE")
            Traceback (most recent call last):
            ...
            agenomic.vault.sensitive.SensitiveUnavailable: environment variable AGENOMIC_DOCTEST_UNSET_VARIABLE is not set
        """
        held = os.environ.get(name)
        if not held:
            raise SensitiveUnavailable(f"environment variable {name} is not set")
        return cls(held)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{MASK}')"

    def __str__(self) -> str:
        return MASK

    def __format__(self, format_spec: str) -> str:
        return MASK

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self) -> tuple[Any, tuple[type[Sensitive]]]:
        return (_masked, (type(self),))

    def __copy__(self) -> Sensitive:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Sensitive:
        return self

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def coerce(value: object) -> Sensitive:
            if isinstance(value, cls):
                return value
            if isinstance(value, str):
                return cls(value)
            raise ValueError("expected a string")

        return core_schema.no_info_plain_validator_function(
            coerce,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _: MASK, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "format": "password", "writeOnly": True}


class IssuedToken(Sensitive):
    """A runtime-identity token the server shows once.

    The plain string can be taken a single time with :meth:`consume`; every
    other rendering is the constant mask.

    Example:
        >>> token = IssuedToken("vrt_example_token_value")
        >>> token.consume()
        'vrt_example_token_value'
        >>> token.consume()
        Traceback (most recent call last):
        ...
        agenomic.vault.sensitive.SensitiveUnavailable: the value was already consumed or is masked
    """

    __slots__ = ()

    def consume(self) -> str:
        """Return the token once, then forget it."""
        held = _unseal(self)
        object.__setattr__(self, "_held", None)
        return held


def _masked(cls: type[Sensitive]) -> Sensitive:
    restored = cls.__new__(cls)
    object.__setattr__(restored, "_held", None)
    return restored


def _unseal(value: Sensitive) -> str:
    held = value._held
    if held is None:
        raise SensitiveUnavailable("the value was already consumed or is masked")
    return held


def _as_sensitive(value: Union[Sensitive, SecretStr]) -> Sensitive:
    if isinstance(value, Sensitive):
        return value
    if isinstance(value, SecretStr):
        return Sensitive(value.get_secret_value())
    raise TypeError(
        f"a secret value must be wrapped in Sensitive(...), not passed as {type(value).__name__}"
    )
