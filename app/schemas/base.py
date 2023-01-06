from types import ModuleType
from typing import Any, ClassVar, Generic, Type, TypeVar, get_origin, get_type_hints

from pydantic import BaseModel, Extra

T = TypeVar("T")


class ObjectConfig(BaseModel, Generic[T]):
    modules: ClassVar[list[ModuleType]]
    name: str

    @property
    def obj_cls(self) -> Type[T]:
        obj_cls: Type[T] | None

        for module in self.__class__.modules:
            obj_cls = getattr(module, self.name, None)
            if obj_cls is not None:
                break

        if obj_cls is None:
            raise ValueError(
                f"Class {self.name} not found from modules: {self.__class__.modules}!"
            )

        return obj_cls

    def to(self) -> T:
        obj: dict = self.dict(exclude={"name"})

        type_hints: dict[str, Type[Any]] = get_type_hints(self.__class__)
        for (field_name, field_type) in type_hints.items():
            if get_origin(field_type) is ClassVar:
                continue

            field_value: Any = getattr(self, field_name)

            # use `ObjectConfig.to` if field is of type `ObjectConfig`
            if issubclass(field_type, ObjectConfig):
                obj[field_name] = field_value.to()

        return self.obj_cls(**obj)

    class Config:
        extra = Extra.allow
