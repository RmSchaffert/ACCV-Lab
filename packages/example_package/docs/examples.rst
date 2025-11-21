Examples
========

This section demonstrates how examples can be included in documentation and how the directory copying system works.

Example Code
------------

The example code is located in the ``examples/`` directory and gets mirrored during the documentation build 
process (due to the ``examples`` directory being listed in the ``docu_referenced_dirs.txt`` file). Here's the 
complete example:

.. note-literalinclude:: ../examples/basic_usage.py
   :language: python
   :caption: Simple usage example
   :name: basic-usage-example

Demonstrating Directory Mirroring
---------------------------------

This documentation can reference the example code because the ``examples/`` directory is mirrored to the 
documentation build location. The ``docu_referenced_dirs.txt`` file specifies which directories to mirror:

.. code-block:: text
   :caption: Directory mirroring configuration
   :name: docu-referenced-dirs

   # This file lists additional directories (besides docs) that are referenced by documentation.
   # The docs directory is always mirrored automatically.
   # Add one directory name per line, without the docs directory.
   # Lines starting with # are comments and are ignored.

   # Example: if your documentation references code in the examples directory, uncomment the lines below
   examples

How Directory Mirroring Works
-----------------------------

1. **Source Location**: The example code lives in ``packages/example_package/examples/``
2. **Build Process**: During documentation build, the ``mirror_referenced_dirs.py`` script mirrors (symlinks 
   by default) the ``examples/`` directory to ``docs/contained_package_docs_mirror/example_package/examples/``
3. **Documentation References**: The documentation can then reference the mirrored files using relative paths 
   like ``../examples/basic_usage.py``
4. **Path Resolution**: The relative paths work because the documentation and examples are now in the same 
   directory with the same structure as in the original source location.

This ensures that documentation references to example code remain functional when mirroring. 