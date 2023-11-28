'use strict';

import crypto from 'crypto';

/**
 * @description Return the given `name` as allowed
 * for use as a path segment in a :class:`URI`,
 * or raise a :class:`KeyError`.
 */
const validate = (name, state = null) => {
  name = String(name);

  if (name.startsWith('$')) {
    name = name.slice(1);
  }

  if (
    !name ||
    name.includes('/') ||
    name.includes('$') ||
    name.includes('<') ||
    name.includes('>') ||
    name.includes(' ')
  ) {
    if (state) {
      throw new Error(`invalid ID for ${state}: ${name}`);
    } else {
      throw new Error(`invalid ID: ${name}`);
    }
  }

  return name;
};

/**
 * An absolute or relative link to a :class:`State`.
 * @class
 * @example
 * new URI("https://example.com/myservice/value_name")
 * new URI("$other_state/method_name")
 * new URI("/state/scalar/value/none")
 */
export default class URI {
  /**
   * @function join
   * @static
   * @description Join the given segments together to make a new URI.
   */
  static join(segments) {
    if (segments?.length > 0) {
      return new URI(...segments);
    } else {
      return new URI('/');
    }
  }

  constructor(subject, ...path) {
    path = path
      .filter((segment) => segment)
      .map((segment) => validate(String(segment)));

    if (subject instanceof URI) {
      this._subject = subject._subject;
      this._path = [...subject._path, ...path];
      return;
    }

    if (typeof subject === 'string') {
      if (subject.startsWith('$$')) {
        throw new Error(`invalid reference: ${subject}`);
      } else if (subject.startsWith('$')) {
        subject = subject.slice(1);
      }
    }

    this._subject = subject;
    this._path = path;

    if (String(this).startsWith('//')) {
      throw new Error(`invalid URI prefix: ${this}`);
    }
  }

  __add__(other) {
    other = String(other);

    if (other.startsWith('$')) {
      throw new Error(`AssertionError: other does not start with '$'`);
    }

    if (other.includes('://')) {
      throw new Error(`ValueError: cannot append ${other} to ${this}`);
    }

    if (!other || other === '/') {
      return this;
    } else if (other.startsWith('/')) {
      other = other.slice(1);
    }

    const path = [...this._path, ...other.split('/')];
    return new URI(this._subject, ...path);
  }

  __radd__(other) {
    return new URI(other).__add__(String(this));
  }

  __bool__() {
    return true;
  }

  __eq__(other) {
    return String(this) === String(other);
  }

  __getitem__(item) {
    const segments = this.split();

    if (typeof item === 'number') {
      if (item === 0) {
        return new URI(segments[0]);
      } else {
        return new URI('/').append(segments[item]);
      }
    } else if (item instanceof Array) {
      if (!segments[item]) {
        return new URI('/');
      }

      if (this.startsWith('$') || '://' in segments[item][0]) {
        return URI.join(segments[item]);
      } else if (segments[item].length === 1) {
        const [segment] = segments[item];

        if ('://' in segment || ['/', '$'].includes(segment[0])) {
          return new URI(segment);
        } else {
          return new URI('/').append(segment);
        }
      } else {
        return new URI('/').extend(...segments[item]);
      }
    } else {
      throw new TypeError(`cannot index a URI with ${item}`);
    }
  }

  __gt__(other) {
    return this.startsWith(other) && this.length > other.length;
  }

  __ge__(other) {
    return this.startsWith(other);
  }

  __len__() {
    return this.split().length;
  }

  __lt__(other) {
    return other.startsWith(this) && this.length < other.length;
  }

  __le__(other) {
    return other.startsWith(this);
  }

  __hash__() {
    const hash = crypto.createHash('sha256');

    hash.update(String(this), 'utf-8');

    const hexHash = hash.digest('hex');

    const decimalHash = BigInt('0x' + hexHash).toString();

    return decimalHash;
  }

  __json__() {
    return { [String(this)]: [] };
  }

  __ns__(cxt, nameHint) {
    // TO DO
    // implementation pending:
    // Scalar ref files implementation pending
  }

  __repr__() {
    if (this._path) {
      return `URI(${this._subject}/${this._path.join('/')})`;
    } else {
      return `URI(${this._subject})`;
    }
  }

  __str__() {
    let root =
      this._subject instanceof URI
        ? String(this._subject.__uri__)
        : String(this._subject);

    if (!(root.startsWith('/') || root.startsWith('$') || '://' in root)) {
      root = `$${root}`;
    }

    const path = this._path.join('/');

    if (path) {
      return root === '/' ? `/${path}` : `${root}/${path}`;
    } else {
      return root;
    }
  }

  /**
   * Construct a new `URI` beginning with this `URI`
   * and ending with the given `suffix`.
   * @example
   * value = OpRef.Get(new URI("http://example.com/myapp").append("value/name"))
   */
  append(suffix) {
    suffix = String(suffix);

    if (suffix === '' || suffix === '/') {
      return this;
    }

    if (suffix.includes('://')) {
      throw new Error(`ValueError: cannot append ${suffix} to ${this}`);
    }

    if (suffix.startsWith('/')) {
      suffix = suffix.slice(1);
    }

    const newPath = [...this._path, ...suffix.split('/')];
    return new URI(this._subject, ...newPath);
  }

  extend(...segments) {
    let validated = [];

    segments.map((segment) => {
      segment = String(segment);

      if (segment.includes('://')) {
        throw new Error(`ValueError: cannot append ${segment} to ${this}`);
      }

      if (segment.startsWith('/')) {
        segment = segment.slice(1);
      }

      if (segment.includes('/')) {
        segment.split('/').forEach((subsegment) => {
          if (validate(subsegment)) {
            validated.push(subsegment);
          }
        });
      } else {
        if (validate(segment)) {
          validated.push(segment);
        }
      }

      return validate(segment);
    });

    return new URI(this._subject, ...this._path, ...validated);
  }

  /**
   * @description
   * Return the ID segment of this `URI`, if present.
   */
  id() {
    const thisStr = String(this);

    if (thisStr.startsWith('$')) {
      const end = thisStr.includes('/') ? thisStr.indexOf('/') : undefined;
      return end ? thisStr.slice(1, end) : thisStr.slice(1);
    }
  }

  /**
   * @description
   * Return `True` if this URI is a simple ID, like `$foo`.
   */
  isId() {
    return !String(this).includes('/');
  }

  /**
   * @description
   * Return the host segment of this `URI`, if present.
   */
  host() {
    const uri = String(this);

    if (!uri.includes('://')) {
      return null;
    }

    const start = uri.indexOf('://') + 3;

    if (!uri.slice(start).includes('/')) {
      return uri.slice(start);
    }

    const end = uri.includes(':', start)
      ? uri.indexOf(':', start)
      : uri.indexOf('/', start);

    return end > start ? uri.slice(start, end) : uri.slice(start);
  }

  /**
   * @description
   * Return the path segment of this `URI`, if present.
   */
  path() {
    const uri = String(this);

    if (!uri.includes('://')) {
      return new URI(uri.slice(uri.indexOf('/')));
    }

    let start = uri.indexOf('://');

    if (!uri.slice(start + 3).includes('/')) {
      return null;
    }

    start = uri.indexOf('/', start + 3);

    return new URI(uri.slice(start));
  }

  /**
   * @description
   * Return the port of this `URI`, if present.
   */
  port() {
    const protocol = this.protocol();

    let prefix = protocol ? protocol + '://' : '';

    const host = this.host();

    prefix += host ? host : '';

    const uri = String(this);

    if (uri === prefix) {
      return null;
    }

    if (prefix && uri[prefix.length] == ':') {
      const end = uri.indexOf('/', prefix.length);
      return parseInt(uri.slice(prefix.length + 1, end));
    }
  }

  /**
   * @description
   * Return the protocol of this `URI` (e.g. "http"), if present.
   */
  protocol() {
    const uri = String(this);

    if (uri.includes('://')) {
      const i = uri.indexOf('://');

      if (i > 0) {
        return uri.slice(0, i);
      }
    }
  }

  /**
   * @description
   * Split this URI into its individual segments.
   * The host (if absolute) or ID (if relative)
   * is treated as the first segment, if present.
   */
  split() {
    if (this.startsWith('/')) {
      const segments = String(this).slice(1).split('/');
      segments[0] = `/${segments[0]}`;

      return segments;
    } else if (this.startsWith('$')) {
      return String(this).split('/');
    } else {
      const host = this.host();
      const port = this.port();

      const calculatedHost = port ? `${host}:${port}` : host;

      return [
        `${this.protocol()}://${calculatedHost}`,
        ...String(this.path()).split('/').slice(1),
      ];
    }
  }

  /**
   * @description
   * Return `True` if this :class:`URI` starts with the given `prefix`.
   */
  startsWith(prefix) {
    return String(this).startsWith(String(prefix));
  }
}
