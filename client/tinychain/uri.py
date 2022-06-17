

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

    def __init__(self, subject, *path):
        if isinstance(subject, URI):
            self._subject = subject._subject
            self._path = subject._path
            self._path += path
            return

        if isinstance(subject, str):
            if subject.startswith("$$"):
                raise ValueError(f"invalid reference: {subject}")
            elif subject.startswith('$'):
                subject = subject[1:]
        else:
            self._subject = subject

        self._subject = subject
        self._path = tuple(str(path_segment) for path_segment in path)

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
        if isinstance(self._subject, str):
            root = self._subject
        elif hasattr(self._subject, "__uri__"):
            root = str(self._subject.__uri__)
        else:
            raise RuntimeError(f"{self._subject} is missing a __uri__ attribute")

        if root.startswith('/') or root.startswith('$') or "://" in root:
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

        path = list(self._path)
        path.append(name)
        return URI(self._subject, *path)

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

        uri = str(self)

        if "://" not in uri:
            return None

        start = uri.index("://") + 3
        if '/' not in uri[start:]:
            return uri[start:]

        end = (
            uri.index(':', start) if ':' in uri[start:]
            else uri.index('/', start))

        if end > start:
            return uri[start:end]
        else:
            return uri[start:]

    def path(self):
        """Return the path segment of this `URI`."""

        uri = str(self)

        if "://" not in uri:
            return uri[uri.index('/'):]

        start = uri.index("://")

        if '/' not in uri[(start + 3):]:
            return None

        start = uri.index('/', start + 3)
        return URI(uri[start:])

    def port(self):
        """Return the port of this `URI`, if present."""

        prefix = self.protocol() + "://" if self.protocol() else ""
        prefix += self.host() if self.host() else ""

        uri = str(self)

        if prefix and uri[len(prefix)] == ':':
            end = uri.index('/', len(prefix))
            return int(uri[(len(prefix) + 1):end])

    def protocol(self):
        """Return the protocol of this `URI` (e.g. "http"), if present."""

        uri = str(self)
        if "://" in uri:
            i = uri.index("://")
            if i > 0:
                return uri[:i]

    def startswith(self, prefix):
        return str(self).startswith(str(prefix))
