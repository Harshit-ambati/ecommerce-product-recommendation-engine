import argparse
import json
from .model import Recommender


def main():
    parser = argparse.ArgumentParser(description='Load a locally trained artifact and recommend products.')
    parser.add_argument('--artifact', default='artifacts/recommender.pkl')
    parser.add_argument('--user', default='U0000')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--method', choices=['popularity', 'content', 'collaborative'])
    args = parser.parse_args()
    model = Recommender.load(args.artifact)
    print(json.dumps({'user_id': args.user, 'method': args.method or model.selected_method,
                      'as_of': str(model.cutoff), 'recommendations': model.recommend(args.user, args.k, args.method)}, indent=2))


if __name__ == '__main__':
    main()
