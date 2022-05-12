

class URI(object):
    """
    An absolute or relative link to a TinyChain state.

    Examples:
        .. highlight:: python
        .. code-block:: python

            URI("https://example.com/myservice/value_name")
            URI("$other_state/method_name")
            URI("/state/scalar/value/none")
    """

    def __init__(self, root, path=[]):
        root = str(root)
        if not root:
            raise ValueError(f"invalid URI root: {root}")

        if root.startswith("$$"):
            raise ValueError(f"invalid reference: {root}")
        elif root.startswith('$'):
            root = root[1:]

        self._root = root
        self._path = path

    def __add__(self, other):
        if other == "/":
            return self
        else:
            return URI(str(self) + other)

    def __radd__(self, other):
        return URI(other) + str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __gt__(self, other):
        return self.startswith(other) and len(self) > len(other)

    def __ge__(self, other):
        return self.startswith(other)

    def __len__(self):
        return len(str(self))

    def __lt__(self, other):
        return other > self

    def __le__(self, other):
        return other >= self

    def __hash__(self):
        return hash(str(self))

    def __json__(self):
        return {str(self): []}

    def __repr__(self):
        return str(self)

    def __str__(self):
        root = self._root
        if root.startswith('/') or "://" in root:
            pass
        else:
            root = f"${root}"

        if self._path:
            path = "/".join(self._path)
            return f"{root}/{path}"
        else:
            return root

    def append(self, name):
        """
        Construct a new `URI` beginning with this `URI` and ending with the given `name` segment.

        Example:
            .. highlight:: python
            .. code-block:: python

                value = OpRef.Get(URI("http://example.com/myapp").append("value_name"))
        """

        if not str(name):
            return self

        return URI(str(self), [str(name)])

    def id(self):
        """Return the ID segment of this `URI`, if present."""

        this = str(self)
        if this.startswith('$'):
            if '/' in this:
                end = this.index("/")
                return this[1:end]
            else:
                return this[1:]

    def is_id(self):
        """Return `True` if this URI is a simple ID, like `$foo`."""

        return '/' not in str(self)

    def host(self):
        """Return the host segment of this `URI`, if present."""

        if "://" not in self._root:
            return None

        start = self._root.index("://") + 3
        if '/' not in self._root[start:]:
            return self._root[start:]

        end = (
            self._root.index(':', start) if ':' in self._root[start:]
            else self._root.index('/', start))

        if end > start:
            return self._root[start:end]
        else:
            return self._root[start:]

    def path(self):
        """Return the path segment of this `URI`."""

        if "://" not in str(self._root):
            return URI(self._root) + "/".join(self._path)

        start = self._root.index("://")

        if '/' not in self._root[(start + 3):]:
            return None

        start = self._root.index('/', start + 3)

        prefix = URI(self._root[start:])
        if self._path:
            return prefix + "/".join(self._path)
        else:
            return prefix

    def port(self):
        """Return the port of this `URI`, if present."""

        prefix = self.protocol() + "://" if self.protocol() else ""
        prefix += self.host() if self.host() else ""
        if prefix and self._root[len(prefix)] == ':':
            end = self._root.index('/', len(prefix))
            return int(self._root[len(prefix) + 1:end])

    def protocol(self):
        """Return the protocol of this `URI` (e.g. "http"), if present."""

        if "://" in self._root:
            i = self._root.index("://")
            if i > 0:
                return self._root[:i]

    def startswith(self, prefix):
        return str(self).startswith(str(prefix))


def uri(subject):
    """Return the `URI` of the given state."""

    if isinstance(subject, URI):
        return subject

    if hasattr(subject, "__uri__"):
        if subject.__uri__ is None:
            raise ValueError(f"{subject} (a {type(subject)}) has a URI of {None}")

        return subject.__uri__

    if hasattr(type(subject), "__uri__"):
        return uri(type(subject))

    return None
