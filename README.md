# facematch

a simple program to recognize face,1:N face match.

|python-version|2.7|
|--|--|
|dlib|19.7|
|lshash|0.04|


How the program runs?

* bulid face datalib:extract landmark and feature on img->generate decriptor on face->save descriptor and name for each img

* build lsh on data and reverse index on lsh:build lsh by Euclidean distance based on descriptor.Besides,build reverse index on lsh's index for search.

* match: first, we search the matched img by lsh. if the result don't meet conditions, we would iter all the descriptors to search.