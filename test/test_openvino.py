from run_infer import parse_args, main


def test_sync_infer_int8():
    cmd = "-t 5 -mt int8 -rm sync"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_sync_infer_fp32():
    cmd = "-t 5 -mt fp32 -rm sync"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_sync_infer_fp16():
    cmd = "-t 5 -mt fp16 -rm sync"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_sync_infer_inference_only():
    cmd = "-t 5 -io -rm sync"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_async_infer():
    cmd = "-t 5 -rm async"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_multi_infer():
    cmd = "-t 5 -rm multi -n 2"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_one_decode_multi_infer():
    cmd = "-t 5 -rm one_decode_multi -n 2"
    test_args = parse_args(cmd.split())
    main(test_args)
