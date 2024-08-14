

def validate(name, state=None):
    """
    Return the given `name` as allowed for use as a path segment in a :class:`URI`, or raise a :class:`KeyError`.
    """

    name = str(name)
    name = str(name[1:]) if name.startswith('$') else name

    if not name or '/' in name or '$' in name or '<' in name or '>' in name or ' ' in name:
        if state:
            raise KeyError(f"invalid ID for {state}: {name}")
        else:
            raise KeyError(f"invalid ID: {name}")

    return name


class URI(object):
    """
    An absolute or relative link to a :class:`State`.

    Examples:
        .. highlight:: python
        .. code-block:: python

            URI("https://example.com/myservice/value_name")
            URI("$other_state/method_name")
            URI("/state/scalar/value/none")
    """

    @staticmethod
    def join(segments):
        """Join the given segments together to make a new URI."""

        if segments:
            return URI(segments[0], *segments[1:])
        else:
            return URI('/')

    def __init__(self, subject, *path):
        path = [validate(segment) for segment in path if str(segment)]

        if isinstance(subject, URI):
            self._subject = subject._subject
            self._path = list(subject._path)
            self._path.extend(path)
            return

        if isinstance(subject, str):
            if subject.startswith("$$"):
                raise ValueError(f"invalid reference: {subject}")
            elif subject.startswith('$'):
                subject = subject[1:]
        else:
            self._subject = subject

        self._subject = subject
        self._path = path

        assert not str(self).startswith("//"), f"invalid URI prefix: {self}"

    def __add__(self, other):
        other = str(other)

        assert not other.startswith('$')

        if other.__contains__("://"):
            raise ValueError(f"cannot append {other} to {self}")

        if not other or other == "/":
            return self
        elif other.startswith('/'):
            other = other[1:]

        path = list(self._path)
        path.extend(other.split('/'))
        return URI(self._subject, *path)

    def __radd__(self, other):
        return URI(other) + str(self)

    def __bool__(self):
        return True

    def __eq__(self, other):
        return str(self) == str(other)

    def __getitem__(self, item):
        segments = self.split()

        if isinstance(item, int):
            return segments[item]
        elif isinstance(item, slice):
            if not segments[item]:
                return URI('/')

            if self.startswith('$'):
                return URI.join(segments[item])
            elif "://" in segments[item][0]:
                return URI.join(segments[item])
            elif len(segments[item]) == 1:
                [segment] = segments[item]
                if "://" in segment or segment[0] in ['/', '$']:
                    return URI(segment)
                else:
                    return URI('/').append(segment)
            else:
                return URI('/').extend(*segments[item])
        else:
            raise TypeError(f"cannot index a URI with {item}")

    def __gt__(self, other):
        return self.startswith(other) and len(self) > len(other)

    def __ge__(self, other):
        return self.startswith(other)

    def __len__(self):
        return len(self.split())

    def __lt__(self, other):
        return other.startswith(self) and len(self) < len(other)

    def __le__(self, other):
        return other.startswith(self)

    def __hash__(self):
        return hash(str(self))

    def __json__(self):
        return {str(self): []}

    def __ns__(self, cxt, name_hint):
        from .scalar.ref import is_literal, is_op_ref
        from .state import State

        cxt.deanonymize(self._subject, name_hint + "_subject")

        if is_op_ref(self._subject):
            cxt.assign(self._subject, name_hint + "_subject")
        elif isinstance(self._subject, State) and is_literal(self._subject):
            cxt.assign(self._subject, name_hint + "_subject")

    def __repr__(self):
        if self._path:
            return f"URI({self._subject}/{'/'.join(self._path)})"
        else:
            return f"URI({self._subject})"

    def __str__(self):
        if hasattr(self._subject, "__uri__"):
            root = str(self._subject.__uri__)
        else:
            root = str(self._subject)

        if root.startswith('/') or root.startswith('$') or "://" in root:
            pass
        else:
            root = f"${root}"

        path = '/'.join(self._path)
        if path:
            if root == '/':
                return f"/{path}"
            else:
                return f"{root}/{path}"
        else:
            return root

    def append(self, suffix):
        """
        Construct a new `URI` beginning with this `URI` and ending with the given `suffix`.

        Example:
            .. highlight:: python
            .. code-block:: python

                value = OpRef.Get(URI("http://example.com/myapp").append("value/name"))
        """

        suffix = str(suffix)

        if suffix in ["", "/"]:
            return self

        if "://" in suffix:
            raise ValueError(f"cannot append {suffix} to {self}")

        if suffix.startswith('/'):
            suffix = suffix[1:]

        path = list(self._path)
        path.extend(suffix.split('/'))

        return URI(self._subject, *path)

    def extend(self, *segments):
        validated = []
        for segment in segments:
            segment = str(segment)

            if "://" in segment:
                raise ValueError(f"cannot append {segment} to {self}")

            if segment.startswith('/'):
                segment = segment[1:]

            if '/' in segment:
                for subsegment in segment.split('/'):
                    validated.append(validate(subsegment))
            else:
                validated.append(validate(segment))

        return URI(self._subject, *list(self._path) + validated)

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
        """Return the path segment of this `URI`, if present."""

        uri = str(self)

        if "://" not in uri:
            return URI(uri[uri.index('/'):])

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

        if uri == prefix:
            return None

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

    def split(self):
        """
        Split this URI into its individual segments.

        The host (if absolute) or ID (if relative) is treated as the first segment, if present.
        """

        if self.startswith('/'):
            return str(self)[1:].split('/')
        elif self.startswith('$'):
            segments = str(self).split('/')
            segments[0] = URI(segments[0])
            return segments
        else:
            host = f"{self.host()}:{self.port()}" if self.port() else self.host()
            return [f"{self.protocol()}://{host}"] + str(self.path()).split('/')[1:]

    def startswith(self, prefix):
        """Return `True` if this :class:`URI` starts with the given `prefix`."""

        return str(self).startswith(str(prefix))
