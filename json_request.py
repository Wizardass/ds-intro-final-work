import dill

import pandas as pd
import json

def main():

    with open('model/sber_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    with open('model/data/json_files/data_753170_277.json') as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        print(f'{y[0]}')

if __name__ == '__main__':
    main()