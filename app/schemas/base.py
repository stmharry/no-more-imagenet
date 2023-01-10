import importlib
from types import ModuleType
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    Type,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
)

from absl import logging
from pydantic import BaseModel, Extra, Field, validator

T = TypeVar("T")
T_factory = Union[Type[T], Callable[..., T]]


class GenericAlias(Protocol):
    __origin__: type[object]


class IndirectGenericSubclass(Protocol):
    __orig_bases__: tuple[GenericAlias]


def is_indirect_generic_subclass(
    obj: object,
) -> TypeGuard[IndirectGenericSubclass]:

    bases = getattr(obj, "__orig_bases__")
    return bases is not None and isinstance(bases, tuple)


class ObjectConfig(BaseModel, Generic[T]):
    obj_cls: T_factory = Field(init=False, alias="__class__", repr=False)

    @validator("obj_cls", pre=True)
    def convert_obj_cls(cls, name: str) -> Any:
        obj_cls: T_factory | None = None

        module_name: str
        cls_name: str
        (module_name, _, obj_name) = name.rpartition(".")

        if module_name == "":
            module_name == "__main__"

        module: ModuleType = importlib.import_module(module_name)
        obj_cls = getattr(module, obj_name, None)

        if obj_cls is None:
            logging.fatal(f"Referenced class {obj_cls} not found!")

        return obj_cls

    def create(self, **kwargs: Any) -> T:
        obj_dict: dict = self.dict()
        logging.info(
            f"Creating object `{self.obj_cls.__name__}` from config {obj_dict}."
        )

        for field_name in obj_dict.keys():
            field_value: Any = getattr(self, field_name)

            # use `ObjectConfig.to` if field is of type `ObjectConfig`
            if isinstance(field_value, ObjectConfig):
                obj_dict[field_name] = field_value.create()

        if kwargs is not None:
            obj_dict.update(kwargs)

        assert is_indirect_generic_subclass(self.__class__)

        obj: T = self.obj_cls(**obj_dict)
        type_T: Type[T] = get_args(self.__class__.__orig_bases__[0])[0]
        if not isinstance(obj, type_T):
            logging.fatal(
                f"Object {obj} is not a sub-class of config-specificed class `{type_T}`!"
            )

        return obj

    # this has to be arranged to the last position to avoid overriding `dict`
    def dict(self, *args, **kwargs) -> dict:
        exclude: set | None = kwargs.pop("exclude", None)
        if exclude is None:
            exclude = set()

        exclude = exclude | {"obj_cls"}

        return super().dict(*args, exclude=exclude, **kwargs)

    class Config:
        extra = Extra.allow
