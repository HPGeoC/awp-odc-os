// Automatically generated; do not edit.
#include <stddef.h>

// 1-D <-> 1-D layout base class.
class Layout_1d {
protected:
  idx_t _d1;

public:

  Layout_1d(idx_t d1) : _d1(d1) { }

  // Return dimension 1.
  virtual idx_t get_d1() const { return _d1; };

  // Return overall number of elements.
  virtual idx_t get_size() const { return _d1; };

  // Return 1-D offset from 1-D 'j' indices.
  virtual idx_t layout(idx_t j1) const =0;

  // Set 1 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1) const =0;
};

// 1-D <-> 1-D layout class with dimensions in d1 order,
// meaning d1 is stored with unit stride.
class Layout_1 : public Layout_1d {
public:

  Layout_1(idx_t d1) : Layout_1d(d1) { }

  // Return 1-D offset from 1-D 'j' indices.
  virtual idx_t layout(idx_t j1) const
    { return j1; }

  // set 1 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1) const
    { j1 = (ai); }
};

// 2-D <-> 1-D layout base class.
class Layout_2d {
protected:
  idx_t _d1, _d2;

public:

  Layout_2d(idx_t d1, idx_t d2) : _d1(d1), _d2(d2) { }

  // Return dimension 1.
  virtual idx_t get_d1() const { return _d1; };

  // Return dimension 2.
  virtual idx_t get_d2() const { return _d2; };

  // Return overall number of elements.
  virtual idx_t get_size() const { return _d1 * _d2; };

  // Return 1-D offset from 2-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2) const =0;

  // Set 2 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2) const =0;
};

// 2-D <-> 1-D layout class with dimensions in d1, d2 order,
// meaning d2 is stored with unit stride.
class Layout_12 : public Layout_2d {
public:

  Layout_12(idx_t d1, idx_t d2) : Layout_2d(d1, d2) { }

  // Return 1-D offset from 2-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2) const
    { return j1 * _d2 + j2; }

  // set 2 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2) const
    { j1 = (ai)/(_d2); j2 = (ai) % _d2; }
};

// 2-D <-> 1-D layout class with dimensions in d2, d1 order,
// meaning d1 is stored with unit stride.
class Layout_21 : public Layout_2d {
public:

  Layout_21(idx_t d1, idx_t d2) : Layout_2d(d1, d2) { }

  // Return 1-D offset from 2-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2) const
    { return j2 * _d1 + j1; }

  // set 2 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2) const
    { j2 = (ai)/(_d1); j1 = (ai) % _d1; }
};

// 3-D <-> 1-D layout base class.
class Layout_3d {
protected:
  idx_t _d1, _d2, _d3;

public:

  Layout_3d(idx_t d1, idx_t d2, idx_t d3) : _d1(d1), _d2(d2), _d3(d3) { }

  // Return dimension 1.
  virtual idx_t get_d1() const { return _d1; };

  // Return dimension 2.
  virtual idx_t get_d2() const { return _d2; };

  // Return dimension 3.
  virtual idx_t get_d3() const { return _d3; };

  // Return overall number of elements.
  virtual idx_t get_size() const { return _d1 * _d2 * _d3; };

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const =0;

  // Set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const =0;
};

// 3-D <-> 1-D layout class with dimensions in d1, d2, d3 order,
// meaning d3 is stored with unit stride.
class Layout_123 : public Layout_3d {
public:

  Layout_123(idx_t d1, idx_t d2, idx_t d3) : Layout_3d(d1, d2, d3) { }

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const
    { return j1 * _d2 * _d3 + j2 * _d3 + j3; }

  // set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const
    { j1 = (ai)/(_d2 * _d3); j2 = (ai)/(_d3) % _d2; j3 = (ai) % _d3; }
};

// 3-D <-> 1-D layout class with dimensions in d1, d3, d2 order,
// meaning d2 is stored with unit stride.
class Layout_132 : public Layout_3d {
public:

  Layout_132(idx_t d1, idx_t d2, idx_t d3) : Layout_3d(d1, d2, d3) { }

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const
    { return j1 * _d3 * _d2 + j3 * _d2 + j2; }

  // set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const
    { j1 = (ai)/(_d3 * _d2); j3 = (ai)/(_d2) % _d3; j2 = (ai) % _d2; }
};

// 3-D <-> 1-D layout class with dimensions in d2, d1, d3 order,
// meaning d3 is stored with unit stride.
class Layout_213 : public Layout_3d {
public:

  Layout_213(idx_t d1, idx_t d2, idx_t d3) : Layout_3d(d1, d2, d3) { }

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const
    { return j2 * _d1 * _d3 + j1 * _d3 + j3; }

  // set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const
    { j2 = (ai)/(_d1 * _d3); j1 = (ai)/(_d3) % _d1; j3 = (ai) % _d3; }
};

// 3-D <-> 1-D layout class with dimensions in d2, d3, d1 order,
// meaning d1 is stored with unit stride.
class Layout_231 : public Layout_3d {
public:

  Layout_231(idx_t d1, idx_t d2, idx_t d3) : Layout_3d(d1, d2, d3) { }

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const
    { return j2 * _d3 * _d1 + j3 * _d1 + j1; }

  // set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const
    { j2 = (ai)/(_d3 * _d1); j3 = (ai)/(_d1) % _d3; j1 = (ai) % _d1; }
};

// 3-D <-> 1-D layout class with dimensions in d3, d1, d2 order,
// meaning d2 is stored with unit stride.
class Layout_312 : public Layout_3d {
public:

  Layout_312(idx_t d1, idx_t d2, idx_t d3) : Layout_3d(d1, d2, d3) { }

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const
    { return j3 * _d1 * _d2 + j1 * _d2 + j2; }

  // set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const
    { j3 = (ai)/(_d1 * _d2); j1 = (ai)/(_d2) % _d1; j2 = (ai) % _d2; }
};

// 3-D <-> 1-D layout class with dimensions in d3, d2, d1 order,
// meaning d1 is stored with unit stride.
class Layout_321 : public Layout_3d {
public:

  Layout_321(idx_t d1, idx_t d2, idx_t d3) : Layout_3d(d1, d2, d3) { }

  // Return 1-D offset from 3-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3) const
    { return j3 * _d2 * _d1 + j2 * _d1 + j1; }

  // set 3 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3) const
    { j3 = (ai)/(_d2 * _d1); j2 = (ai)/(_d1) % _d2; j1 = (ai) % _d1; }
};

// 4-D <-> 1-D layout base class.
class Layout_4d {
protected:
  idx_t _d1, _d2, _d3, _d4;

public:

  Layout_4d(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : _d1(d1), _d2(d2), _d3(d3), _d4(d4) { }

  // Return dimension 1.
  virtual idx_t get_d1() const { return _d1; };

  // Return dimension 2.
  virtual idx_t get_d2() const { return _d2; };

  // Return dimension 3.
  virtual idx_t get_d3() const { return _d3; };

  // Return dimension 4.
  virtual idx_t get_d4() const { return _d4; };

  // Return overall number of elements.
  virtual idx_t get_size() const { return _d1 * _d2 * _d3 * _d4; };

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const =0;

  // Set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const =0;
};

// 4-D <-> 1-D layout class with dimensions in d1, d2, d3, d4 order,
// meaning d4 is stored with unit stride.
class Layout_1234 : public Layout_4d {
public:

  Layout_1234(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j1 * _d2 * _d3 * _d4 + j2 * _d3 * _d4 + j3 * _d4 + j4; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j1 = (ai)/(_d2 * _d3 * _d4); j2 = (ai)/(_d3 * _d4) % _d2; j3 = (ai)/(_d4) % _d3; j4 = (ai) % _d4; }
};

// 4-D <-> 1-D layout class with dimensions in d1, d2, d4, d3 order,
// meaning d3 is stored with unit stride.
class Layout_1243 : public Layout_4d {
public:

  Layout_1243(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j1 * _d2 * _d4 * _d3 + j2 * _d4 * _d3 + j4 * _d3 + j3; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j1 = (ai)/(_d2 * _d4 * _d3); j2 = (ai)/(_d4 * _d3) % _d2; j4 = (ai)/(_d3) % _d4; j3 = (ai) % _d3; }
};

// 4-D <-> 1-D layout class with dimensions in d1, d3, d2, d4 order,
// meaning d4 is stored with unit stride.
class Layout_1324 : public Layout_4d {
public:

  Layout_1324(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j1 * _d3 * _d2 * _d4 + j3 * _d2 * _d4 + j2 * _d4 + j4; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j1 = (ai)/(_d3 * _d2 * _d4); j3 = (ai)/(_d2 * _d4) % _d3; j2 = (ai)/(_d4) % _d2; j4 = (ai) % _d4; }
};

// 4-D <-> 1-D layout class with dimensions in d1, d3, d4, d2 order,
// meaning d2 is stored with unit stride.
class Layout_1342 : public Layout_4d {
public:

  Layout_1342(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j1 * _d3 * _d4 * _d2 + j3 * _d4 * _d2 + j4 * _d2 + j2; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j1 = (ai)/(_d3 * _d4 * _d2); j3 = (ai)/(_d4 * _d2) % _d3; j4 = (ai)/(_d2) % _d4; j2 = (ai) % _d2; }
};

// 4-D <-> 1-D layout class with dimensions in d1, d4, d2, d3 order,
// meaning d3 is stored with unit stride.
class Layout_1423 : public Layout_4d {
public:

  Layout_1423(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j1 * _d4 * _d2 * _d3 + j4 * _d2 * _d3 + j2 * _d3 + j3; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j1 = (ai)/(_d4 * _d2 * _d3); j4 = (ai)/(_d2 * _d3) % _d4; j2 = (ai)/(_d3) % _d2; j3 = (ai) % _d3; }
};

// 4-D <-> 1-D layout class with dimensions in d1, d4, d3, d2 order,
// meaning d2 is stored with unit stride.
class Layout_1432 : public Layout_4d {
public:

  Layout_1432(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j1 * _d4 * _d3 * _d2 + j4 * _d3 * _d2 + j3 * _d2 + j2; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j1 = (ai)/(_d4 * _d3 * _d2); j4 = (ai)/(_d3 * _d2) % _d4; j3 = (ai)/(_d2) % _d3; j2 = (ai) % _d2; }
};

// 4-D <-> 1-D layout class with dimensions in d2, d1, d3, d4 order,
// meaning d4 is stored with unit stride.
class Layout_2134 : public Layout_4d {
public:

  Layout_2134(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j2 * _d1 * _d3 * _d4 + j1 * _d3 * _d4 + j3 * _d4 + j4; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j2 = (ai)/(_d1 * _d3 * _d4); j1 = (ai)/(_d3 * _d4) % _d1; j3 = (ai)/(_d4) % _d3; j4 = (ai) % _d4; }
};

// 4-D <-> 1-D layout class with dimensions in d2, d1, d4, d3 order,
// meaning d3 is stored with unit stride.
class Layout_2143 : public Layout_4d {
public:

  Layout_2143(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j2 * _d1 * _d4 * _d3 + j1 * _d4 * _d3 + j4 * _d3 + j3; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j2 = (ai)/(_d1 * _d4 * _d3); j1 = (ai)/(_d4 * _d3) % _d1; j4 = (ai)/(_d3) % _d4; j3 = (ai) % _d3; }
};

// 4-D <-> 1-D layout class with dimensions in d2, d3, d1, d4 order,
// meaning d4 is stored with unit stride.
class Layout_2314 : public Layout_4d {
public:

  Layout_2314(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j2 * _d3 * _d1 * _d4 + j3 * _d1 * _d4 + j1 * _d4 + j4; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j2 = (ai)/(_d3 * _d1 * _d4); j3 = (ai)/(_d1 * _d4) % _d3; j1 = (ai)/(_d4) % _d1; j4 = (ai) % _d4; }
};

// 4-D <-> 1-D layout class with dimensions in d2, d3, d4, d1 order,
// meaning d1 is stored with unit stride.
class Layout_2341 : public Layout_4d {
public:

  Layout_2341(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j2 * _d3 * _d4 * _d1 + j3 * _d4 * _d1 + j4 * _d1 + j1; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j2 = (ai)/(_d3 * _d4 * _d1); j3 = (ai)/(_d4 * _d1) % _d3; j4 = (ai)/(_d1) % _d4; j1 = (ai) % _d1; }
};

// 4-D <-> 1-D layout class with dimensions in d2, d4, d1, d3 order,
// meaning d3 is stored with unit stride.
class Layout_2413 : public Layout_4d {
public:

  Layout_2413(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j2 * _d4 * _d1 * _d3 + j4 * _d1 * _d3 + j1 * _d3 + j3; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j2 = (ai)/(_d4 * _d1 * _d3); j4 = (ai)/(_d1 * _d3) % _d4; j1 = (ai)/(_d3) % _d1; j3 = (ai) % _d3; }
};

// 4-D <-> 1-D layout class with dimensions in d2, d4, d3, d1 order,
// meaning d1 is stored with unit stride.
class Layout_2431 : public Layout_4d {
public:

  Layout_2431(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j2 * _d4 * _d3 * _d1 + j4 * _d3 * _d1 + j3 * _d1 + j1; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j2 = (ai)/(_d4 * _d3 * _d1); j4 = (ai)/(_d3 * _d1) % _d4; j3 = (ai)/(_d1) % _d3; j1 = (ai) % _d1; }
};

// 4-D <-> 1-D layout class with dimensions in d3, d1, d2, d4 order,
// meaning d4 is stored with unit stride.
class Layout_3124 : public Layout_4d {
public:

  Layout_3124(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j3 * _d1 * _d2 * _d4 + j1 * _d2 * _d4 + j2 * _d4 + j4; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j3 = (ai)/(_d1 * _d2 * _d4); j1 = (ai)/(_d2 * _d4) % _d1; j2 = (ai)/(_d4) % _d2; j4 = (ai) % _d4; }
};

// 4-D <-> 1-D layout class with dimensions in d3, d1, d4, d2 order,
// meaning d2 is stored with unit stride.
class Layout_3142 : public Layout_4d {
public:

  Layout_3142(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j3 * _d1 * _d4 * _d2 + j1 * _d4 * _d2 + j4 * _d2 + j2; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j3 = (ai)/(_d1 * _d4 * _d2); j1 = (ai)/(_d4 * _d2) % _d1; j4 = (ai)/(_d2) % _d4; j2 = (ai) % _d2; }
};

// 4-D <-> 1-D layout class with dimensions in d3, d2, d1, d4 order,
// meaning d4 is stored with unit stride.
class Layout_3214 : public Layout_4d {
public:

  Layout_3214(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j3 * _d2 * _d1 * _d4 + j2 * _d1 * _d4 + j1 * _d4 + j4; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j3 = (ai)/(_d2 * _d1 * _d4); j2 = (ai)/(_d1 * _d4) % _d2; j1 = (ai)/(_d4) % _d1; j4 = (ai) % _d4; }
};

// 4-D <-> 1-D layout class with dimensions in d3, d2, d4, d1 order,
// meaning d1 is stored with unit stride.
class Layout_3241 : public Layout_4d {
public:

  Layout_3241(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j3 * _d2 * _d4 * _d1 + j2 * _d4 * _d1 + j4 * _d1 + j1; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j3 = (ai)/(_d2 * _d4 * _d1); j2 = (ai)/(_d4 * _d1) % _d2; j4 = (ai)/(_d1) % _d4; j1 = (ai) % _d1; }
};

// 4-D <-> 1-D layout class with dimensions in d3, d4, d1, d2 order,
// meaning d2 is stored with unit stride.
class Layout_3412 : public Layout_4d {
public:

  Layout_3412(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j3 * _d4 * _d1 * _d2 + j4 * _d1 * _d2 + j1 * _d2 + j2; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j3 = (ai)/(_d4 * _d1 * _d2); j4 = (ai)/(_d1 * _d2) % _d4; j1 = (ai)/(_d2) % _d1; j2 = (ai) % _d2; }
};

// 4-D <-> 1-D layout class with dimensions in d3, d4, d2, d1 order,
// meaning d1 is stored with unit stride.
class Layout_3421 : public Layout_4d {
public:

  Layout_3421(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j3 * _d4 * _d2 * _d1 + j4 * _d2 * _d1 + j2 * _d1 + j1; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j3 = (ai)/(_d4 * _d2 * _d1); j4 = (ai)/(_d2 * _d1) % _d4; j2 = (ai)/(_d1) % _d2; j1 = (ai) % _d1; }
};

// 4-D <-> 1-D layout class with dimensions in d4, d1, d2, d3 order,
// meaning d3 is stored with unit stride.
class Layout_4123 : public Layout_4d {
public:

  Layout_4123(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j4 * _d1 * _d2 * _d3 + j1 * _d2 * _d3 + j2 * _d3 + j3; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j4 = (ai)/(_d1 * _d2 * _d3); j1 = (ai)/(_d2 * _d3) % _d1; j2 = (ai)/(_d3) % _d2; j3 = (ai) % _d3; }
};

// 4-D <-> 1-D layout class with dimensions in d4, d1, d3, d2 order,
// meaning d2 is stored with unit stride.
class Layout_4132 : public Layout_4d {
public:

  Layout_4132(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j4 * _d1 * _d3 * _d2 + j1 * _d3 * _d2 + j3 * _d2 + j2; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j4 = (ai)/(_d1 * _d3 * _d2); j1 = (ai)/(_d3 * _d2) % _d1; j3 = (ai)/(_d2) % _d3; j2 = (ai) % _d2; }
};

// 4-D <-> 1-D layout class with dimensions in d4, d2, d1, d3 order,
// meaning d3 is stored with unit stride.
class Layout_4213 : public Layout_4d {
public:

  Layout_4213(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j4 * _d2 * _d1 * _d3 + j2 * _d1 * _d3 + j1 * _d3 + j3; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j4 = (ai)/(_d2 * _d1 * _d3); j2 = (ai)/(_d1 * _d3) % _d2; j1 = (ai)/(_d3) % _d1; j3 = (ai) % _d3; }
};

// 4-D <-> 1-D layout class with dimensions in d4, d2, d3, d1 order,
// meaning d1 is stored with unit stride.
class Layout_4231 : public Layout_4d {
public:

  Layout_4231(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j4 * _d2 * _d3 * _d1 + j2 * _d3 * _d1 + j3 * _d1 + j1; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j4 = (ai)/(_d2 * _d3 * _d1); j2 = (ai)/(_d3 * _d1) % _d2; j3 = (ai)/(_d1) % _d3; j1 = (ai) % _d1; }
};

// 4-D <-> 1-D layout class with dimensions in d4, d3, d1, d2 order,
// meaning d2 is stored with unit stride.
class Layout_4312 : public Layout_4d {
public:

  Layout_4312(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j4 * _d3 * _d1 * _d2 + j3 * _d1 * _d2 + j1 * _d2 + j2; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j4 = (ai)/(_d3 * _d1 * _d2); j3 = (ai)/(_d1 * _d2) % _d3; j1 = (ai)/(_d2) % _d1; j2 = (ai) % _d2; }
};

// 4-D <-> 1-D layout class with dimensions in d4, d3, d2, d1 order,
// meaning d1 is stored with unit stride.
class Layout_4321 : public Layout_4d {
public:

  Layout_4321(idx_t d1, idx_t d2, idx_t d3, idx_t d4) : Layout_4d(d1, d2, d3, d4) { }

  // Return 1-D offset from 4-D 'j' indices.
  virtual idx_t layout(idx_t j1, idx_t j2, idx_t j3, idx_t j4) const
    { return j4 * _d3 * _d2 * _d1 + j3 * _d2 * _d1 + j2 * _d1 + j1; }

  // set 4 'j' indices based on 1-D 'ai' input.
  virtual void unlayout(idx_t ai, idx_t& j1, idx_t& j2, idx_t& j3, idx_t& j4) const
    { j4 = (ai)/(_d3 * _d2 * _d1); j3 = (ai)/(_d2 * _d1) % _d3; j2 = (ai)/(_d1) % _d2; j1 = (ai) % _d1; }
};
