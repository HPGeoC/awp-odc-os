// Automatically generated; do not edit.

// 1-D <-> 1-D layout macros.
// 'LAYOUT' macros return 1-D offset from 1-D 'j' indices.
// 'UNLAYOUT' macros set 1 'j' indices based on 1-D 'ai' input.
#define LAYOUT_1(j1, d1) ((j1))
#define UNLAYOUT_1(ai, j1, d1) (j1 = (ai))

// 2-D <-> 1-D layout macros.
// 'LAYOUT' macros return 1-D offset from 2-D 'j' indices.
// 'UNLAYOUT' macros set 2 'j' indices based on 1-D 'ai' input.
#define LAYOUT_12(j1, j2, d1, d2) ((j1) * (d2) + (j2))
#define UNLAYOUT_12(ai, j1, j2, d1, d2) (j1 = (ai)/((d2)), j2 = (ai) % (d2))
#define LAYOUT_21(j1, j2, d1, d2) ((j2) * (d1) + (j1))
#define UNLAYOUT_21(ai, j1, j2, d1, d2) (j2 = (ai)/((d1)), j1 = (ai) % (d1))

// 3-D <-> 1-D layout macros.
// 'LAYOUT' macros return 1-D offset from 3-D 'j' indices.
// 'UNLAYOUT' macros set 3 'j' indices based on 1-D 'ai' input.
#define LAYOUT_123(j1, j2, j3, d1, d2, d3) ((j1) * (d2) * (d3) + (j2) * (d3) + (j3))
#define UNLAYOUT_123(ai, j1, j2, j3, d1, d2, d3) (j1 = (ai)/((d2) * (d3)), j2 = (ai)/((d3)) % (d2), j3 = (ai) % (d3))
#define LAYOUT_132(j1, j2, j3, d1, d2, d3) ((j1) * (d3) * (d2) + (j3) * (d2) + (j2))
#define UNLAYOUT_132(ai, j1, j2, j3, d1, d2, d3) (j1 = (ai)/((d3) * (d2)), j3 = (ai)/((d2)) % (d3), j2 = (ai) % (d2))
#define LAYOUT_213(j1, j2, j3, d1, d2, d3) ((j2) * (d1) * (d3) + (j1) * (d3) + (j3))
#define UNLAYOUT_213(ai, j1, j2, j3, d1, d2, d3) (j2 = (ai)/((d1) * (d3)), j1 = (ai)/((d3)) % (d1), j3 = (ai) % (d3))
#define LAYOUT_231(j1, j2, j3, d1, d2, d3) ((j2) * (d3) * (d1) + (j3) * (d1) + (j1))
#define UNLAYOUT_231(ai, j1, j2, j3, d1, d2, d3) (j2 = (ai)/((d3) * (d1)), j3 = (ai)/((d1)) % (d3), j1 = (ai) % (d1))
#define LAYOUT_312(j1, j2, j3, d1, d2, d3) ((j3) * (d1) * (d2) + (j1) * (d2) + (j2))
#define UNLAYOUT_312(ai, j1, j2, j3, d1, d2, d3) (j3 = (ai)/((d1) * (d2)), j1 = (ai)/((d2)) % (d1), j2 = (ai) % (d2))
#define LAYOUT_321(j1, j2, j3, d1, d2, d3) ((j3) * (d2) * (d1) + (j2) * (d1) + (j1))
#define UNLAYOUT_321(ai, j1, j2, j3, d1, d2, d3) (j3 = (ai)/((d2) * (d1)), j2 = (ai)/((d1)) % (d2), j1 = (ai) % (d1))

// 4-D <-> 1-D layout macros.
// 'LAYOUT' macros return 1-D offset from 4-D 'j' indices.
// 'UNLAYOUT' macros set 4 'j' indices based on 1-D 'ai' input.
#define LAYOUT_1234(j1, j2, j3, j4, d1, d2, d3, d4) ((j1) * (d2) * (d3) * (d4) + (j2) * (d3) * (d4) + (j3) * (d4) + (j4))
#define UNLAYOUT_1234(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j1 = (ai)/((d2) * (d3) * (d4)), j2 = (ai)/((d3) * (d4)) % (d2), j3 = (ai)/((d4)) % (d3), j4 = (ai) % (d4))
#define LAYOUT_1243(j1, j2, j3, j4, d1, d2, d3, d4) ((j1) * (d2) * (d4) * (d3) + (j2) * (d4) * (d3) + (j4) * (d3) + (j3))
#define UNLAYOUT_1243(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j1 = (ai)/((d2) * (d4) * (d3)), j2 = (ai)/((d4) * (d3)) % (d2), j4 = (ai)/((d3)) % (d4), j3 = (ai) % (d3))
#define LAYOUT_1324(j1, j2, j3, j4, d1, d2, d3, d4) ((j1) * (d3) * (d2) * (d4) + (j3) * (d2) * (d4) + (j2) * (d4) + (j4))
#define UNLAYOUT_1324(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j1 = (ai)/((d3) * (d2) * (d4)), j3 = (ai)/((d2) * (d4)) % (d3), j2 = (ai)/((d4)) % (d2), j4 = (ai) % (d4))
#define LAYOUT_1342(j1, j2, j3, j4, d1, d2, d3, d4) ((j1) * (d3) * (d4) * (d2) + (j3) * (d4) * (d2) + (j4) * (d2) + (j2))
#define UNLAYOUT_1342(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j1 = (ai)/((d3) * (d4) * (d2)), j3 = (ai)/((d4) * (d2)) % (d3), j4 = (ai)/((d2)) % (d4), j2 = (ai) % (d2))
#define LAYOUT_1423(j1, j2, j3, j4, d1, d2, d3, d4) ((j1) * (d4) * (d2) * (d3) + (j4) * (d2) * (d3) + (j2) * (d3) + (j3))
#define UNLAYOUT_1423(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j1 = (ai)/((d4) * (d2) * (d3)), j4 = (ai)/((d2) * (d3)) % (d4), j2 = (ai)/((d3)) % (d2), j3 = (ai) % (d3))
#define LAYOUT_1432(j1, j2, j3, j4, d1, d2, d3, d4) ((j1) * (d4) * (d3) * (d2) + (j4) * (d3) * (d2) + (j3) * (d2) + (j2))
#define UNLAYOUT_1432(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j1 = (ai)/((d4) * (d3) * (d2)), j4 = (ai)/((d3) * (d2)) % (d4), j3 = (ai)/((d2)) % (d3), j2 = (ai) % (d2))
#define LAYOUT_2134(j1, j2, j3, j4, d1, d2, d3, d4) ((j2) * (d1) * (d3) * (d4) + (j1) * (d3) * (d4) + (j3) * (d4) + (j4))
#define UNLAYOUT_2134(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j2 = (ai)/((d1) * (d3) * (d4)), j1 = (ai)/((d3) * (d4)) % (d1), j3 = (ai)/((d4)) % (d3), j4 = (ai) % (d4))
#define LAYOUT_2143(j1, j2, j3, j4, d1, d2, d3, d4) ((j2) * (d1) * (d4) * (d3) + (j1) * (d4) * (d3) + (j4) * (d3) + (j3))
#define UNLAYOUT_2143(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j2 = (ai)/((d1) * (d4) * (d3)), j1 = (ai)/((d4) * (d3)) % (d1), j4 = (ai)/((d3)) % (d4), j3 = (ai) % (d3))
#define LAYOUT_2314(j1, j2, j3, j4, d1, d2, d3, d4) ((j2) * (d3) * (d1) * (d4) + (j3) * (d1) * (d4) + (j1) * (d4) + (j4))
#define UNLAYOUT_2314(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j2 = (ai)/((d3) * (d1) * (d4)), j3 = (ai)/((d1) * (d4)) % (d3), j1 = (ai)/((d4)) % (d1), j4 = (ai) % (d4))
#define LAYOUT_2341(j1, j2, j3, j4, d1, d2, d3, d4) ((j2) * (d3) * (d4) * (d1) + (j3) * (d4) * (d1) + (j4) * (d1) + (j1))
#define UNLAYOUT_2341(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j2 = (ai)/((d3) * (d4) * (d1)), j3 = (ai)/((d4) * (d1)) % (d3), j4 = (ai)/((d1)) % (d4), j1 = (ai) % (d1))
#define LAYOUT_2413(j1, j2, j3, j4, d1, d2, d3, d4) ((j2) * (d4) * (d1) * (d3) + (j4) * (d1) * (d3) + (j1) * (d3) + (j3))
#define UNLAYOUT_2413(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j2 = (ai)/((d4) * (d1) * (d3)), j4 = (ai)/((d1) * (d3)) % (d4), j1 = (ai)/((d3)) % (d1), j3 = (ai) % (d3))
#define LAYOUT_2431(j1, j2, j3, j4, d1, d2, d3, d4) ((j2) * (d4) * (d3) * (d1) + (j4) * (d3) * (d1) + (j3) * (d1) + (j1))
#define UNLAYOUT_2431(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j2 = (ai)/((d4) * (d3) * (d1)), j4 = (ai)/((d3) * (d1)) % (d4), j3 = (ai)/((d1)) % (d3), j1 = (ai) % (d1))
#define LAYOUT_3124(j1, j2, j3, j4, d1, d2, d3, d4) ((j3) * (d1) * (d2) * (d4) + (j1) * (d2) * (d4) + (j2) * (d4) + (j4))
#define UNLAYOUT_3124(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j3 = (ai)/((d1) * (d2) * (d4)), j1 = (ai)/((d2) * (d4)) % (d1), j2 = (ai)/((d4)) % (d2), j4 = (ai) % (d4))
#define LAYOUT_3142(j1, j2, j3, j4, d1, d2, d3, d4) ((j3) * (d1) * (d4) * (d2) + (j1) * (d4) * (d2) + (j4) * (d2) + (j2))
#define UNLAYOUT_3142(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j3 = (ai)/((d1) * (d4) * (d2)), j1 = (ai)/((d4) * (d2)) % (d1), j4 = (ai)/((d2)) % (d4), j2 = (ai) % (d2))
#define LAYOUT_3214(j1, j2, j3, j4, d1, d2, d3, d4) ((j3) * (d2) * (d1) * (d4) + (j2) * (d1) * (d4) + (j1) * (d4) + (j4))
#define UNLAYOUT_3214(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j3 = (ai)/((d2) * (d1) * (d4)), j2 = (ai)/((d1) * (d4)) % (d2), j1 = (ai)/((d4)) % (d1), j4 = (ai) % (d4))
#define LAYOUT_3241(j1, j2, j3, j4, d1, d2, d3, d4) ((j3) * (d2) * (d4) * (d1) + (j2) * (d4) * (d1) + (j4) * (d1) + (j1))
#define UNLAYOUT_3241(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j3 = (ai)/((d2) * (d4) * (d1)), j2 = (ai)/((d4) * (d1)) % (d2), j4 = (ai)/((d1)) % (d4), j1 = (ai) % (d1))
#define LAYOUT_3412(j1, j2, j3, j4, d1, d2, d3, d4) ((j3) * (d4) * (d1) * (d2) + (j4) * (d1) * (d2) + (j1) * (d2) + (j2))
#define UNLAYOUT_3412(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j3 = (ai)/((d4) * (d1) * (d2)), j4 = (ai)/((d1) * (d2)) % (d4), j1 = (ai)/((d2)) % (d1), j2 = (ai) % (d2))
#define LAYOUT_3421(j1, j2, j3, j4, d1, d2, d3, d4) ((j3) * (d4) * (d2) * (d1) + (j4) * (d2) * (d1) + (j2) * (d1) + (j1))
#define UNLAYOUT_3421(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j3 = (ai)/((d4) * (d2) * (d1)), j4 = (ai)/((d2) * (d1)) % (d4), j2 = (ai)/((d1)) % (d2), j1 = (ai) % (d1))
#define LAYOUT_4123(j1, j2, j3, j4, d1, d2, d3, d4) ((j4) * (d1) * (d2) * (d3) + (j1) * (d2) * (d3) + (j2) * (d3) + (j3))
#define UNLAYOUT_4123(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j4 = (ai)/((d1) * (d2) * (d3)), j1 = (ai)/((d2) * (d3)) % (d1), j2 = (ai)/((d3)) % (d2), j3 = (ai) % (d3))
#define LAYOUT_4132(j1, j2, j3, j4, d1, d2, d3, d4) ((j4) * (d1) * (d3) * (d2) + (j1) * (d3) * (d2) + (j3) * (d2) + (j2))
#define UNLAYOUT_4132(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j4 = (ai)/((d1) * (d3) * (d2)), j1 = (ai)/((d3) * (d2)) % (d1), j3 = (ai)/((d2)) % (d3), j2 = (ai) % (d2))
#define LAYOUT_4213(j1, j2, j3, j4, d1, d2, d3, d4) ((j4) * (d2) * (d1) * (d3) + (j2) * (d1) * (d3) + (j1) * (d3) + (j3))
#define UNLAYOUT_4213(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j4 = (ai)/((d2) * (d1) * (d3)), j2 = (ai)/((d1) * (d3)) % (d2), j1 = (ai)/((d3)) % (d1), j3 = (ai) % (d3))
#define LAYOUT_4231(j1, j2, j3, j4, d1, d2, d3, d4) ((j4) * (d2) * (d3) * (d1) + (j2) * (d3) * (d1) + (j3) * (d1) + (j1))
#define UNLAYOUT_4231(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j4 = (ai)/((d2) * (d3) * (d1)), j2 = (ai)/((d3) * (d1)) % (d2), j3 = (ai)/((d1)) % (d3), j1 = (ai) % (d1))
#define LAYOUT_4312(j1, j2, j3, j4, d1, d2, d3, d4) ((j4) * (d3) * (d1) * (d2) + (j3) * (d1) * (d2) + (j1) * (d2) + (j2))
#define UNLAYOUT_4312(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j4 = (ai)/((d3) * (d1) * (d2)), j3 = (ai)/((d1) * (d2)) % (d3), j1 = (ai)/((d2)) % (d1), j2 = (ai) % (d2))
#define LAYOUT_4321(j1, j2, j3, j4, d1, d2, d3, d4) ((j4) * (d3) * (d2) * (d1) + (j3) * (d2) * (d1) + (j2) * (d1) + (j1))
#define UNLAYOUT_4321(ai, j1, j2, j3, j4, d1, d2, d3, d4) (j4 = (ai)/((d3) * (d2) * (d1)), j3 = (ai)/((d2) * (d1)) % (d3), j2 = (ai)/((d1)) % (d2), j1 = (ai) % (d1))
