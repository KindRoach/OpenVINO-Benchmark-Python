from exp import dynamic_shape, ov_preprocess, auto_batch


def test_dynamic_shape():
    dynamic_shape.main()


def test_ov_preprocess():
    ov_preprocess.main()

def test_auto_batch():
    auto_batch.main()
