#!/usr/bin/env python3
"""Generate the API reference of the RT-COSMIK book from the docstrings.

The sources are read statically with griffe: nothing is imported, so building
the documentation needs neither torch nor Pinocchio nor acados. One Markdown
page is written per area of the code, under docs/api/, where SUMMARY.md lists
them.

    python3 docs/gen_api.py                       # from the repository root
    python3 docs/gen_api.py --src SRC --out OUT   # explicit paths
"""
import argparse
import logging
import re
from pathlib import Path

import griffe

REPO_URL = "https://github.com/Gepetto/rt-cosmik/blob/main"

#: The reference, one page per area: file, title, introduction, modules.
PAGES = [
    ("camera.md", "Cameras and calibration",
     "Loading a calibration, finding the attached cameras, and reading them.",
     ["camera.cam_utils", "camera.camera", "camera.sources"]),
    ("nlf.md", "Pose estimation",
     "The default front end: person detection and NLF's metric 3D landmarks, per view.",
     ["nlf.nlf"]),
    ("triangulation.md", "Multi-view fusion",
     "Turning the landmarks of every view into one set of 3D points.",
     ["triangulation.triangulation"]),
    ("filtering.md", "Filtering",
     "The causal low-pass filter applied to the fused landmarks.",
     ["filtering.iir"]),
    ("pipeline.md", "Pipeline and solver",
     "The calibration and inverse kinematics of one person, and the live pipeline process.",
     ["pipeline.solver", "pipeline.pipeline"]),
    ("ik.md", "Inverse kinematics",
     "The per-frame and moving-horizon solvers, and the generated optimal control problem.",
     ["ik.ik", "ik.ocp_model"]),
    ("human_model.md", "Human model",
     "Scaling the model to a person and registering the landmarks on its segments.",
     ["human_model.model_utils"]),
    ("ergonomics.md", "Ergonomics",
     "REBA scoring.",
     ["ergonomics.reba"]),
    ("saver.md", "Recording",
     "Saving landmarks and joint angles, and the recording keys.",
     ["saver.recorder", "saver.csv_saver", "saver.hotkeys"]),
    ("viewer.md", "Viewer",
     "The 3D viewer and the scene of the COMFI dataset.",
     ["viewer.viewer", "viewer.comfi_scene", "viewer.async_display"]),
    ("utils.md", "Utilities",
     "Datasets, shared memory, video reading, and helpers.",
     ["utils.dataset", "utils.mp_utils", "utils.VideoReader", "utils.read_write_utils",
      "utils.linear_algebra_utils", "utils.ergo_utils"]),
    ("config.md", "Configuration and models",
     "Loading `settings.py`, and finding the model files the configuration names.",
     ["config_loader", "model_weights"]),
]

_ROLE = re.compile(r":(?:class|func|meth|mod|attr|data|obj|exc):`(~?)([^`]+)`")


def markdown(text):
    """reStructuredText habits of the docstrings, written as Markdown."""
    text = _ROLE.sub(lambda m: f"`{m.group(2).split('.')[-1] if m.group(1) else m.group(2)}`", text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    # Placeholders such as camera_<id> would read as HTML tags; escape them
    # outside inline code, where they are already literal.
    text = "`".join(part if i % 2 else re.sub(r"<(\w[^<>]*)>", r"&lt;\1&gt;", part)
                    for i, part in enumerate(text.split("`")))
    # "Typical use::" introduces an indented literal block, which Markdown
    # already renders as code; drop the second colon.
    return re.sub(r"::\n", ":\n", text)


def annotation(expr):
    return f"`{expr}`" if expr is not None else ""


def item_text(description):
    """A docstring item's description, indented to stay inside its list item."""
    return markdown(description).strip().replace("\n", "\n  ")


def signature(name, function, drop_self=False):
    parts = []
    for parameter in function.parameters:
        if drop_self and parameter.name in ("self", "cls"):
            continue
        piece = parameter.name
        if parameter.kind is griffe.ParameterKind.var_positional:
            piece = "*" + piece
        elif parameter.kind is griffe.ParameterKind.var_keyword:
            piece = "**" + piece
        if parameter.annotation is not None:
            piece += f": {parameter.annotation}"
            if parameter.default is not None:
                piece += f" = {parameter.default}"
        elif parameter.default is not None:
            piece += f"={parameter.default}"
        parts.append(piece)
    one_line = f"{name}({', '.join(parts)})"
    if len(one_line) <= 88:
        return one_line
    return f"{name}(\n" + "".join(f"    {p},\n" for p in parts) + ")"


def docstring(obj):
    """Markdown for an object's docstring, section by section."""
    if obj.docstring is None:
        return ""
    out = []
    for section in obj.docstring.parsed:
        kind = section.kind.value
        if kind == "text":
            out.append(markdown(section.value))
        elif kind in ("parameters", "other parameters", "attributes"):
            title = {"parameters": "Parameters", "other parameters": "Other parameters",
                     "attributes": "Attributes"}[kind]
            lines = [f"**{title}:**", ""]
            for item in section.value:
                kind_of = f" ({annotation(item.annotation)})" if item.annotation is not None else ""
                lines.append(f"- `{item.name}`{kind_of}: {item_text(item.description)}")
            out.append("\n".join(lines))
        elif kind in ("returns", "yields"):
            title = "Returns" if kind == "returns" else "Yields"
            lines = [f"**{title}:**", ""]
            for item in section.value:
                head = annotation(item.annotation) or (f"`{item.name}`" if item.name else "")
                description = item_text(item.description)
                lines.append(f"- {head}: {description}" if head else f"- {description}")
            out.append("\n".join(lines))
        elif kind in ("raises", "warns"):
            title = "Raises" if kind == "raises" else "Warns"
            lines = [f"**{title}:**", ""]
            for item in section.value:
                lines.append(f"- {annotation(item.annotation)}: {item_text(item.description)}")
            out.append("\n".join(lines))
        elif kind == "examples":
            for sub_kind, value in section.value:
                if sub_kind.value == "examples":
                    out.append(f"```python\n{value}\n```")
                else:
                    out.append(markdown(value))
        elif kind == "admonition":
            out.append(f"**{section.title or 'Note'}:** {markdown(section.value.description)}")
        else:
            out.append(markdown(str(section.value)))
    return "\n\n".join(part for part in out if part.strip())


def source_link(obj, src_root):
    path = Path(obj.filepath).resolve().relative_to(src_root.parent.resolve())
    return f"[source]({REPO_URL}/{path.as_posix()}#L{obj.lineno})"


def is_documented_member(name, member):
    """Public objects defined here, not names imported from elsewhere."""
    if member.is_alias:
        return False
    if name.startswith("_"):
        return name == "__call__"
    return member.is_function or member.is_class or (member.is_attribute and member.docstring)


def render_function(name, function, src_root, level, method=False):
    lines = [f"{'#' * level} `{name}`", "",
             f"```python\n{signature(name, function, drop_self=method)}\n```", ""]
    body = docstring(function)
    if body:
        lines += [body, ""]
    lines += [source_link(function, src_root), ""]
    return lines


def render_class(name, cls, src_root, level):
    init = cls.members.get("__init__")
    lines = [f"{'#' * level} `{name}`", ""]
    if init is not None and init.is_function and not init.is_alias:
        lines += [f"```python\nclass {signature(name, init, drop_self=True)}\n```", ""]
    else:
        lines += [f"```python\nclass {name}\n```", ""]
    body = docstring(cls)
    init_body = docstring(init) if init is not None and not init.is_alias else ""
    for part in (body, init_body):
        if part:
            lines += [part, ""]
    lines += [source_link(cls, src_root), ""]
    for member_name, member in sorted(cls.members.items(), key=lambda kv: kv[1].lineno or 0):
        if member_name == "__init__" or not is_documented_member(member_name, member):
            continue
        if member.is_function:
            lines += render_function(member_name, member, src_root, level + 1, method=True)
        elif member.is_attribute and member.docstring:
            lines += [f"{'#' * (level + 1)} `{member_name}`", "", docstring(member), ""]
    return lines


def render_module(module, src_root):
    dotted = module.path
    lines = [f"## `{dotted}`", ""]
    body = docstring(module)
    if body:
        lines += [body, ""]
    members = [(n, m) for n, m in module.members.items() if is_documented_member(n, m)]
    for name, member in sorted(members, key=lambda kv: kv[1].lineno or 0):
        if member.is_class:
            lines += render_class(name, member, src_root, 3)
        elif member.is_function:
            lines += render_function(name, member, src_root, 3)
        elif member.is_attribute:
            lines += [f"### `{name}`", "", docstring(member), ""]
    return lines


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=here.parent / "src",
                        help="directory holding the rtcosmik package")
    parser.add_argument("--out", type=Path, default=here / "api",
                        help="where to write the pages")
    args = parser.parse_args()

    logging.getLogger("griffe").setLevel(logging.ERROR)
    package = griffe.load("rtcosmik", search_paths=[str(args.src)],
                          docstring_parser="google", resolve_aliases=False)
    args.out.mkdir(parents=True, exist_ok=True)

    index = ["# API reference", "",
             "Generated from the docstrings of the `rtcosmik` package. Each entry links",
             "to its source on GitHub.", ""]
    for filename, title, intro, modules in PAGES:
        lines = [f"# {title}", "", intro, ""]
        for dotted in modules:
            lines += render_module(package[dotted], args.src)
        (args.out / filename).write_text("\n".join(lines).rstrip() + "\n")
        index.append(f"- [{title}]({filename}): {intro}")
    (args.out / "README.md").write_text("\n".join(index) + "\n")
    print(f"{len(PAGES)} pages written to {args.out}")


if __name__ == "__main__":
    main()
