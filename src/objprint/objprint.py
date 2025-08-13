# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/objprint/blob/master/NOTICE.txt

import inspect
import json
import re
import itertools
from types import FrameType, FunctionType
from typing import Any, Set, Optional, List, Callable, Type, Tuple, NamedTuple

# Assuming _PrintConfig and color utilities are defined as in the original code
class _PrintConfig:
    # Original configuration logic preserved
    def __init__(self):
        self.enable = True
        self.depth = 100
        self.indent = 2
        self.width = 80
        self.elements = -1
        self.color = True
        self.line_number = False
        self.arg_name = False
        self.skip_recursion = True
        self.honor_existing = True
        self.attr_pattern = r"[^_].*"  # Default: include non-private attributes
        self.include = []
        self.exclude = []
        self.print_methods = False
        self.label = []

    def overwrite(self, **kwargs):
        new_cfg = _PrintConfig()
        new_cfg.__dict__.update(self.__dict__)
        new_cfg.__dict__.update(kwargs)
        return new_cfg

    def set(self,** kwargs):
        self.__dict__.update(kwargs)

class COLOR:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    END = "\033[0m"

def set_color(s: str, color: str) -> str:
    return f"{color}{s}{COLOR.END}"


class ObjPrint:
    FormatterInfo = NamedTuple('FormatterInfo', [('formatter', Callable), ('inherit', bool)])

    def __init__(self):
        self._configs = _PrintConfig()
        self.indicator_map = {
            list: "[]",
            tuple: "()",
            dict: "{}",
            set: "{}"
        }
        self._sys_print = print
        self.frame_analyzer = FrameAnalyzer()  # Use modified FrameAnalyzer
        self.type_formatter = {}

    def __call__(self, *objs: Any, file: Any = None, format: str = "string", **kwargs) -> Any:
        cfg = self._configs.overwrite(** kwargs)
        if cfg.enable:
            call_frame = inspect.currentframe()
            if call_frame is not None:
                call_frame = call_frame.f_back

            kwargs.pop("arg_name", None)

            if cfg.line_number:
                self._sys_print(self._get_line_number_str(call_frame, cfg=cfg))

            if cfg.arg_name:
                args = self.frame_analyzer.get_args(call_frame)
                if args is None:
                    args = ["Unknown Arg" for _ in range(len(objs))]
                if cfg.color:
                    args = [set_color(f"{arg}:", COLOR.RED) for arg in args]
                else:
                    args = [f"{arg}:" for arg in args]

            if format == "json":
                if cfg.arg_name:
                    for arg, obj in zip(args, objs):
                        self._sys_print(arg)
                        self._sys_print(json.dumps(self.objjson(obj), **kwargs))
                else:
                    for obj in objs:
                        self._sys_print(json.dumps(self.objjson(obj),** kwargs))
            else:
                kwargs["color"] = cfg.color
                if cfg.arg_name:
                    for arg, obj in zip(args, objs):
                        self._sys_print(arg)
                        self._sys_print(self.objstr(obj, **kwargs), file=file)
                else:
                    for obj in objs:
                        self._sys_print(self.objstr(obj,** kwargs), file=file)
            if self.frame_analyzer.return_object(call_frame):
                return objs[0] if len(objs) == 1 else objs
            else:
                return None

        return objs[0] if len(objs) == 1 else objs

    # Rest of the original ObjPrint methods (objstr, _objstr, objjson, etc.) preserved
    def objstr(self, obj: Any, **kwargs) -> str:
        if "color" not in kwargs:
            kwargs["color"] = False
        cfg = self._configs.overwrite(** kwargs)
        memo: Optional[Set[int]] = set() if cfg.skip_recursion else None
        return self._objstr(obj, memo, indent_level=0, cfg=cfg)

    def _objstr(self, obj: Any, memo: Optional[Set[int]], indent_level: int, cfg: _PrintConfig) -> str:
        # Original logic for formatting objects preserved
        if self.type_formatter:
            obj_type = type(obj)
            for cls in obj_type.__mro__:
                if cls in self.type_formatter and (
                    cls == obj_type or self.type_formatter[cls].inherit
                ):
                    return self.type_formatter[cls].formatter(obj)

        if isinstance(obj, str):
            return f"'{obj}'"
        elif isinstance(obj, (int, float)) or obj is None:
            return str(obj)
        elif isinstance(obj, FunctionType):
            return f"<function {obj.__name__}>"

        if (memo is not None and id(obj) in memo) or \
                (cfg.depth is not None and indent_level >= cfg.depth):
            return self._get_ellipsis(obj, cfg)

        if memo is not None:
            memo = memo.copy()
            memo.add(id(obj))

        if isinstance(obj, (list, tuple, set)):
            elems = (f"{self._objstr(val, memo, indent_level + 1, cfg)}" for val in obj)
        elif isinstance(obj, dict):
            items = [(key, val) for key, val in obj.items()]
            try:
                items = sorted(items)
            except TypeError:
                pass
            elems = (
                f"{self._objstr(key, None, indent_level + 1, cfg)}: {self._objstr(val, memo, indent_level + 1, cfg)}"
                for key, val in items
            )
        else:
            if cfg.honor_existing and \
                    (obj.__class__.__str__ is not object.__str__ or obj.__class__.__repr__ is not object.__repr__):
                s = str(obj)
                lines = s.split("\n")
                lines[1:] = [self.add_indent(line, indent_level, cfg) for line in lines[1:]]
                return "\n".join(lines)
            return self._get_custom_object_str(obj, memo, indent_level, cfg)

        return self._get_pack_str(elems, obj, indent_level, cfg)

    def objjson(self, obj: Any) -> Any:
        return self._objjson(obj, set())

    def _objjson(self, obj: Any, memo: Set[int]) -> Any:
        """
        return a jsonifiable object from obj
        """
        if isinstance(obj, (str, int, float)) or obj is None:
            return obj

        if id(obj) in memo:
            raise ValueError("Can't jsonify a recursive object")

        memo.add(id(obj))

        if isinstance(obj, (list, tuple)):
            return [self._objjson(elem, memo.copy()) for elem in obj]

        if isinstance(obj, dict):
            return {key: self._objjson(val, memo.copy()) for key, val in obj.items()}

        # For generic object
        ret = {".type": type(obj).__name__}

        if hasattr(obj, "__dict__"):
            for key, val in obj.__dict__.items():
                ret[key] = self._objjson(val, memo.copy())

        return ret

    def _get_custom_object_str(self, obj: Any, memo: Optional[Set[int]], indent_level: int, cfg: _PrintConfig):

        def _get_method_line(attr: str) -> str:
            try:
                method_sig = str(inspect.signature(getattr(obj, attr)))
            except ValueError:
                # Please consider special handling
                method_sig = "(<signature unknown>)"

            if cfg.color:
                return f"{set_color('def', COLOR.MAGENTA)} "\
                    f"{set_color(attr, COLOR.GREEN)}{method_sig}"
            else:
                return f"def {attr}{method_sig}"

        def _get_line(key: str) -> str:
            val = self._objstr(getattr(obj, key), memo, indent_level + 1, cfg)
            if cfg.label and any(re.fullmatch(pattern, key) is not None for pattern in cfg.label):
                return set_color(f".{key} = {val}", COLOR.YELLOW)
            elif cfg.color:
                return f"{set_color('.' + key, COLOR.GREEN)} = {val}"
            else:
                return f".{key} = {val}"

        attrs = []
        methods = []
        for attr in dir(obj):
            if re.fullmatch(cfg.attr_pattern, attr):
                if cfg.include:
                    if not any((re.fullmatch(pattern, attr) is not None for pattern in cfg.include)):
                        continue
                if cfg.exclude:
                    if any((re.fullmatch(pattern, attr) is not None for pattern in cfg.exclude)):
                        continue

                try:
                    attr_val = getattr(obj, attr)
                except AttributeError:
                    continue

                if inspect.ismethod(attr_val) or inspect.isbuiltin(attr_val):
                    if cfg.print_methods:
                        methods.append(attr)
                else:
                    attrs.append(attr)

        elems = itertools.chain(
            (_get_method_line(attr) for attr in sorted(methods)),
            (_get_line(key) for key in sorted(attrs))
        )

        return self._get_pack_str(elems, obj, indent_level, cfg)

    def _get_line_number_str(self, curr_frame: Optional[FrameType], cfg: _PrintConfig):
        if curr_frame is None:
            return "Unknown Line Number"
        curr_code = curr_frame.f_code
        if cfg.color:
            return f"{set_color(curr_code.co_name, COLOR.GREEN)} ({curr_code.co_filename}:{curr_frame.f_lineno})"
        else:
            return f"{curr_code.co_name} ({curr_code.co_filename}:{curr_frame.f_lineno})"

    def enable(self) -> None:
        self.config(enable=True)

    def disable(self) -> None:
        self.config(enable=False)

    def config(self, **kwargs) -> None:
        self._configs.set(**kwargs)

    def install(self, name: str = "op") -> None:
        import builtins
        builtins.__dict__[name] = self

    def add_indent(
            self,
            line: SourceLine,
            indent_level: int,
            cfg: _PrintConfig) -> SourceLine:
        if isinstance(line, str):
            return " " * (indent_level * cfg.indent) + line
        return [" " * (indent_level * cfg.indent) + ll for ll in line]

    def register_formatter(
        self,
        obj_type: Type[Any],
        obj_formatter: Optional[Callable[[Any], str]] = None,
        inherit: bool = True
    ) -> Optional[Callable[[Callable[[Any], str]], Callable[[Any], str]]]:
        if obj_formatter is None:
            def wrapper(obj_formatter: Callable[[Any], str]) -> Callable[[Any], str]:
                self.register_formatter(obj_type, obj_formatter, inherit)
                return obj_formatter
            return wrapper

        if not isinstance(obj_type, type):
            raise TypeError("obj_type must be a type")

        if not callable(obj_formatter):
            raise TypeError("obj_formatter must be a callable")

        fmt_info = self.FormatterInfo(formatter=obj_formatter, inherit=inherit)
        self.type_formatter[obj_type] = fmt_info
        return None

    def unregister_formatter(self, *obj_types: Type[Any]) -> None:
        if not obj_types:
            self.type_formatter.clear()
        else:
            for obj_type in obj_types:
                if obj_type in self.type_formatter:
                    del self.type_formatter[obj_type]

    def get_formatter(self) -> dict:
        return self.type_formatter

    def _get_header_footer(self, obj: Any, cfg: _PrintConfig):
        obj_type = type(obj)
        if obj_type in self.indicator_map:
            indicator = self.indicator_map[obj_type]
            return indicator[0], indicator[1]
        else:
            if cfg.color:
                return set_color(f"<{obj_type.__name__} {hex(id(obj))}", COLOR.CYAN), set_color(">", COLOR.CYAN)
            else:
                return f"<{obj_type.__name__} {hex(id(obj))}", ">"

    def _get_ellipsis(self, obj: Any, cfg: _PrintConfig) -> str:
        header, footer = self._get_header_footer(obj, cfg)
        return f"{header} ... {footer}"

    def _get_pack_str(
            self,
            elems: Iterable[str],
            obj: Any,
            indent_level: int,
            cfg: _PrintConfig) -> str:
        """
        :param elems generator: generator of string elements to pack together
        :param obj_type type: object type
        :param indent_level int: current indent level
        """
        header, footer = self._get_header_footer(obj, cfg)

        if cfg.elements == -1:
            elems = list(elems)
        else:
            first_elems = []
            it = iter(elems)
            try:
                for _ in range(cfg.elements):
                    first_elems.append(next(it))
            except StopIteration:
                pass
            if next(it, None) is not None:
                first_elems.append("...")
            elems = first_elems

        multiline = False
        if len(header) > 1 and len(elems) > 0:
            # If it's not built in, always do multiline
            multiline = True
        elif any(("\n" in elem for elem in elems)):
            # Has \n, need multiple mode
            multiline = True
        elif cfg.width is not None and sum((len(elem) for elem in elems)) > cfg.width:
            multiline = True

        if multiline:
            s = ",\n".join(self.add_indent(elems, indent_level + 1, cfg))
            return f"{header}\n{s}\n{self.add_indent('', indent_level, cfg)}{footer}"
        else:
            s = ", ".join(elems)
            return f"{header}{s}{footer}"
