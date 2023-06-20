from run_decode import parse_args, main


def test_sync_decode():
    cmd = "-t 5 -rm sync"
    test_args = parse_args(cmd.split())
    main(test_args)


def test_multi_decode():
    cmd = "-t 5 -rm multi -n 2"
    test_args = parse_args(cmd.split())
    main(test_args)
