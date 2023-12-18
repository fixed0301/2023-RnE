import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=int, help='1/2/3')
    parser.add_argument('--data', type=int, help='csv on which 1031/1214')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('--min_data_len', type=int, default=45)
    return parser.parse_args()

def main():
    args = get_arguments()
    print('\n------------------------Loading Model------------------------\n')


if __name__ == "__main__":
    main()
