Geographer Development
======================

## Repositories
Our main repository is called _geographer_ and on Github: [github.com/kit-parco/geographer](https://github.com/kit-parco/geographer)
It is open to the public and continuously tested with Travis at \url{https://travis-ci.org/kit-parco/geographer}.

Currently there is a private repository called _geographer-dev_, which contains the current content of _geographer_ and, in addition, a few branches of unfinished development. The old _ParcoRepart_ repository is incompatible and should not be used any more.

## Developing
For bugfixes or development which can be public, create a new branch in the _geographer_ repository and push your changes to it.
If you do not have write access, create a public fork instead.
Then, create a pull request on github and assign someone to review the code. It should only be merged if the unit tests in Travis run without errors. 

For private developments, for example implementations for future papers that should not be public yet, create a private fork of the main geographer repository and develop there.
Regularly merge changes from the main repository into your fork to avoid large merge conflicts in the end.
To do this, call `git pull https://github.com/kit-parco/geographer` in your local repository.

## Publishing Changes
There are two ways to make your changes public:
- Change the status of your entire repository to public, then create a pull request towards _geographer/Dev_ on Github.
- Locally clone _geographer_, create a new branch and check it out, call `git pull <your-repo>` to pull in your changes. Then, push the new branch and create a pull request to the _Dev_ branch.