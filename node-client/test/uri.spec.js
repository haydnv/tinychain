'use strict';

import URI from '../tinychain/uri';

describe('Class URI Test', () => {
  describe('append method tests', () => {
    it('returns original uri if no suffix given', () => {
      const uri = new URI('http://example.com/myapp');

      const newUri = uri.append();

      expect(newUri._subject).toBe(uri._subject);
      expect(newUri._path).toBe(uri._path);
    });

    it('returns original uri if / is given as suffix', () => {
      const uri = new URI('http://example.com/myapp');

      const newUri = uri.append('/');

      expect(newUri._subject).toBe(uri._subject);
      expect(newUri._path).toBe(uri._path);
    });

    it('generates error if :// is given as suffix', () => {
      const uri = new URI('http://example.com/myapp');

      expect(() => uri.append('://')).toThrow(
        `ValueError: cannot append :// to ${uri}`,
      );
    });

    it('returns the new appended uri if suffix given', () => {
      const uri = new URI('http://example.com/myapp');

      const newUri = uri.append('/hello');

      let expectedPath = [...uri._path];
      expectedPath.push('hello');

      expect(newUri._subject).toBe(`${uri._subject}`);
      expect(newUri._path).toStrictEqual(expectedPath);
    });
  });

  describe('extend method tests', () => {
    it('returns the new uri with extended path', () => {
      const uri = new URI('http://example.com/myapp');

      const newUri = uri.extend('/hello', 'world');

      let expectedPath = [...uri._path];
      expectedPath.push('hello');
      expectedPath.push('world');

      expect(newUri._subject).toBe(`${uri._subject}`);
      expect(newUri._path).toStrictEqual(expectedPath);
    });

    it('generates error if :// is passed as an argument', () => {
      const uri = new URI('http://example.com/myapp');

      expect(() => uri.extend('/hello', 'world', '://')).toThrow(
        `ValueError: cannot append :// to ${uri}`,
      );
    });

    it('returns the new uri with extended path but passed substrings', () => {
      const uri = new URI('http://example.com/myapp');

      const newUri = uri.extend('/hello/world', 'hello/me');

      let expectedPath = [...uri._path];
      expectedPath.push('hello');
      expectedPath.push('world');
      expectedPath.push('hello');
      expectedPath.push('me');

      expect(newUri._subject).toBe(`${uri._subject}`);
      expect(newUri._path).toStrictEqual(expectedPath);
    });
  });

  describe('id method tests', () => {
    it('returns the id if starts with $', () => {
      const uri = new URI('$123/profile');

      const foundId = uri.id();

      expect(foundId).toBe(`123`);
    });

    it('returns undefined if uri does not start with $', () => {
      const uri = new URI('http://example.com/myapp');

      const foundId = uri.id();

      expect(foundId).toBeUndefined();
    });
  });

  describe('isId method tests', () => {
    it('returns true as it simply an id', () => {
      const uri = new URI('$123');

      const foundId = uri.isId();

      expect(foundId).toBeTruthy();
    });

    it('returns false as it is not a simple id', () => {
      const uri = new URI('$123/profile');

      const foundId = uri.isId();

      expect(foundId).toBeFalsy();
    });
  });

  describe('host method tests', () => {
    it('returns host of the uri if no port is present', () => {
      const uri = new URI('http://example.com/myapp');

      const host = uri.host();

      expect(host).toBe(`example.com`);
    });

    it('returns host of the uri if port is present', () => {
      const uri = new URI('http://example.com:5000');

      const host = uri.host();

      expect(host).toBe(`example.com`);
    });

    it('returns null if no protocol is present', () => {
      const uri = new URI('example.com/myapp');

      const host = uri.host();

      expect(host).toBeNull();
    });
  });

  describe('path method tests', () => {
    it('returns path of the uri if present and no :// found', () => {
      const uri = new URI('example.com/myapp/123/profile');

      const path = uri.path();

      expect(path._subject).toBe(`/myapp/123/profile`);
    });

    it('returns path of the uri if present and :// found', () => {
      const uri = new URI('http://example.com/myapp/123/profile');

      const path = uri.path();

      expect(path._subject).toBe(`/myapp/123/profile`);
    });

    it('returns null if path does not exist', () => {
      const uri = new URI('http://example.com');

      const path = uri.path();

      expect(path).toBeNull();
    });
  });

  describe('port method tests', () => {
    it('returns port of the uri without path if present', () => {
      const uri = new URI('http://example.com:5000');

      const port = uri.port();

      expect(port).toBe(5000);
    });

    it('returns port of the uri with path if present', () => {
      const uri = new URI('http://example.com:5000/myapp');

      const port = uri.port();

      expect(port).toBe(5000);
    });
  });

  describe('protocol method tests', () => {
    it('returns protocol of the uri if present', () => {
      const uri = new URI('http://example.com:5000');

      const protocol = uri.protocol();

      expect(protocol).toBe('http');
    });

    it('returns undefined if protocol not present', () => {
      const uri = new URI('example.com:5000');

      const protocol = uri.protocol();

      expect(protocol).toBeUndefined();
    });
  });

  describe('split method tests', () => {
    it('returns segments of the uri when protocol and host are present', () => {
      const uri = new URI('http://example.com:5000/myapp/123/profile');

      const split = uri.split();

      expect(split).toStrictEqual([
        'http://example.com:5000',
        'myapp',
        '123',
        'profile',
      ]);
    });

    it('returns segments of the uri when it starts with /', () => {
      const uri = new URI('/myapp/123/profile');

      const split = uri.split();

      expect(split).toStrictEqual(['/myapp', '123', 'profile']);
    });

    it('returns segments of the uri when it starts with $', () => {
      const uri = new URI('$myapp/123/profile');

      const split = uri.split();

      expect(split).toStrictEqual(['$myapp', '123', 'profile']);
    });
  });

  describe('startsWith method tests', () => {
    it('returns true if the uri starts with prefix', () => {
      const uri = new URI('http://example.com:5000/myapp/123/profile');

      expect(uri.startsWith('http')).toBeTruthy();
    });

    it('returns false if the uri does not starts with prefix', () => {
      const uri = new URI('example.com:5000/myapp/123/profile');

      expect(uri.startsWith('http')).toBeFalsy();
    });
  });
});
