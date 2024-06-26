# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import sys
import tempfile
import unittest

from iree.compiler.tools.ir_tool import __main__ as ir_tool

# TODO: No idea why pytype cannot find names from this module.
# pytype: disable=name-error
import iree.compiler.tools.tflite

if not iree.compiler.tools.tflite.is_available():
    print(
        f"Skipping test {__file__} because the IREE TFLite compiler "
        f"is not installed"
    )
    sys.exit(0)


def mlir_bytecode_file_to_text(bytecode_file):
    with tempfile.NamedTemporaryFile() as temp_file:
        args = ir_tool.parse_arguments(["copy", bytecode_file, "-o", temp_file.name])
        ir_tool.main(args)
        return str(temp_file.read())


def mlir_bytecode_to_text(bytecode):
    with tempfile.NamedTemporaryFile("wb") as temp_bytecode_file:
        temp_bytecode_file.write(bytecode)
        temp_bytecode_file.flush()
        return mlir_bytecode_file_to_text(temp_bytecode_file.name)


class CompilerTest(unittest.TestCase):
    def testImportBinaryPbFile(self):
        path = os.path.join(os.path.dirname(__file__), "testdata", "tflite_sample.fb")
        bytecode = iree.compiler.tools.tflite.compile_file(path, import_only=True)
        text = mlir_bytecode_to_text(bytecode)
        logging.info("%s", text)
        self.assertIn("tosa.mul", text)

    def testCompileBinaryPbFile(self):
        path = os.path.join(os.path.dirname(__file__), "testdata", "tflite_sample.fb")
        binary = iree.compiler.tools.tflite.compile_file(
            path, target_backends=iree.compiler.tools.tflite.DEFAULT_TESTING_BACKENDS
        )
        logging.info("Binary length = %d", len(binary))
        self.assertIn(b"main", binary)

    def testImportBinaryPbFileOutputFile(self):
        path = os.path.join(os.path.dirname(__file__), "testdata", "tflite_sample.fb")
        with tempfile.NamedTemporaryFile("wt", delete=False) as f:
            try:
                f.close()
                output = iree.compiler.tools.tflite.compile_file(
                    path, import_only=True, output_file=f.name
                )
                self.assertIsNone(output)
                with open(f.name, "rb") as f_read:
                    bytecode = f_read.read()
            finally:
                os.remove(f.name)
        text = mlir_bytecode_to_text(bytecode)
        logging.info("%s", text)
        self.assertIn("tosa.mul", text)

    def testCompileBinaryPbFileOutputFile(self):
        path = os.path.join(os.path.dirname(__file__), "testdata", "tflite_sample.fb")
        with tempfile.NamedTemporaryFile("wt", delete=False) as f:
            try:
                f.close()
                output = iree.compiler.tools.tflite.compile_file(
                    path,
                    output_file=f.name,
                    target_backends=iree.compiler.tools.tflite.DEFAULT_TESTING_BACKENDS,
                )
                self.assertIsNone(output)
                with open(f.name, "rb") as f_read:
                    binary = f_read.read()
            finally:
                os.remove(f.name)
        logging.info("Binary length = %d", len(binary))
        self.assertIn(b"main", binary)

    def testImportBinaryPbBytes(self):
        path = os.path.join(os.path.dirname(__file__), "testdata", "tflite_sample.fb")
        with open(path, "rb") as f:
            content = f.read()
        bytecode = iree.compiler.tools.tflite.compile_str(content, import_only=True)
        text = mlir_bytecode_to_text(bytecode)
        logging.info("%s", text)
        self.assertIn("tosa.mul", text)

    def testCompileBinaryPbBytes(self):
        path = os.path.join(os.path.dirname(__file__), "testdata", "tflite_sample.fb")
        with open(path, "rb") as f:
            content = f.read()
        binary = iree.compiler.tools.tflite.compile_str(
            content, target_backends=iree.compiler.tools.tflite.DEFAULT_TESTING_BACKENDS
        )
        logging.info("Binary length = %d", len(binary))
        self.assertIn(b"main", binary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
