from DataScrutctures import *


Now, this can be pretty annoying, but the solution is very simple. Python identifies a module as a single file, and the assumption that leads to the issue above is that we assume a package has the same convention of just being a directory

However, a packgage requires an __init__.py file to be defined so that they are recognized correctly

So, we need to add two files:

package1/__init__.py
package2/subpackage1/__init__.py
Additionally, these files are completely empty, and only serve as information. Note that we don't need to include a file in package2 directly as this directory does not contain any modules within it so is not really a package in itself and is simply a wrapper subpackage1


Now, this can be pretty annoying, but the solution is very simple. Python identifies a module as a single file, and the assumption that leads to the issue above is that we assume a package has the same convention of just being a directory

However, a packgage requires an __init__.py file to be defined so that they are recognized correctly

So, we need to add two files:

package1/__init__.py
package2/subpackage1/__init__.py
Additionally, these files are completely empty, and only serve as information. Note that we don't need to include a file in package2 directly as this directory does not contain any modules within it so is not really a package in itself and is simply a wrapper subpackage1

https://nabeelvalley.co.za/stdout/2021/06-05/multi-module-python-projects/