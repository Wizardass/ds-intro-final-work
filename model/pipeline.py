def main():
    import dill
    import pickle
    import pandas as pd
    import datetime
    import time
    import logging

    from geopy.geocoders import Nominatim
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer, make_column_selector

    def df_union(df, df2):
        df_copy_1 = df.copy()
        df_copy_2 = df2.copy()

        values_to_select = [
            'sub_car_claim_click',
            'sub_car_claim_submit_click',
            'sub_open_dialog_click',
            'sub_custom_question_submit_click',
            'sub_call_number_click',
            'sub_callback_submit_click',
            'sub_submit_success',
            'sub_car_request_submit_click'
        ]

        selected_rows = df_copy_2[df_copy_2['event_action'].isin(values_to_select)][['session_id', 'event_action']]
        selected_rows['event_action'] = 1
        selected_rows = selected_rows.drop_duplicates()
        df_copy_1 = df_copy_1.merge(selected_rows, on='session_id', how='left')
        df_copy_1['event_action'] = df_copy_1['event_action'].apply(lambda x: 0 if pd.isnull(x) else x)
        df_copy_1['event_action'] = df_copy_1['event_action'].astype(int)

        return df_copy_1.drop(columns=['device_model', 'utm_keyword'])

    def lower_str(data):
        df_copy = data.copy()

        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].str.lower()

        return df_copy

    def set_device_brand(data):
        df_copy = data.copy()

        def set_brand(x):
            import pandas as pd

            if pd.notnull(x['device_brand']):
                return x['device_brand']
            elif x['device_browser'] == 'samsung internet':
                return 'samsung'
            elif x['device_browser'] == 'safari':
                return 'apple'
            elif (x['device_category'] == 'desktop') or (
                    int(x['device_screen_resolution'].split('x')[0]) > int(
                x['device_screen_resolution'].split('x')[1])):
                return 'pc'
            else:
                return '(not set)'

        df_copy['device_brand'] = df_copy.apply(set_brand, axis=1)
        return df_copy

    def clean_device_os(data):
        df_copy = data.copy()

        def set_os(x):
            import pandas as pd

            brand = x['device_brand']
            if pd.isna(x['device_os']):
                if isinstance(brand, str):
                    if 'apple' in brand:
                        return 'ios'
                    elif 'pc' in brand:
                        return 'windows'
                return 'android'
            return x['device_os']

        df_copy['device_os'] = df_copy.apply(set_os, axis=1)
        return df_copy

    def fill_other(data):
        df_copy = data.copy()
        return df_copy.fillna('other')

    def set_date(data):
        df_copy = data.copy()
        import pandas as pd

        df_copy['visit_date'] = pd.to_datetime(df_copy['visit_date'])
        df_copy['month'] = df_copy.visit_date.apply(lambda x: x.month)
        df_copy['dayofweek'] = df_copy.visit_date.apply(lambda x: x.day)
        df_copy['visit_time'] = pd.to_datetime(df_copy['visit_time'], format='%H:%M:%S')
        df_copy['hour'] = df_copy['visit_time'].dt.hour

        return df_copy.drop(columns=['visit_date', 'visit_time'])

    def dev_screen(data):
        df_copy = data.copy()

        df_copy['dev_scr_res_len'] = df_copy.apply(lambda x: len(x.device_screen_resolution), axis=1)
        return df_copy.drop(columns=['device_screen_resolution'])

    def device_browser(data):
        df_copy = data.copy()

        def clean_device_browser(browser):
            if 'instagram' in browser:
                return 'instagram'
            elif 'android' in browser:
                return 'android webview'
            elif 'opera' in browser:
                return 'opera'
            elif 'mozilla' in browser:
                return 'mozilla'
            elif 'internet explorer' in browser:
                return 'edge'
            else:
                return browser

        df_copy['device_browser'] = df_copy['device_browser'].apply(clean_device_browser)
        return df_copy

    def is_russia(data):
        df_copy = data.copy()

        df_copy['is_russia'] = df_copy['geo_country'].apply(lambda x: 1 if x == 'russia' else 0)

        return df_copy

    def is_available(data):
        df_copy = data.copy()

        df_copy['is_available'] = df_copy['geo_city'].apply(
            lambda x: 1 if x == 'moscow' or x == 'saint petersburg' else 0)

        return df_copy

    def clean_geo_data(data):
        df_copy = data.copy()

        df_copy['geo_country'] = df_copy['geo_country'].apply(lambda x: 'россия' if x == 'russia' else x)

        cities = {
            'tuymazy': 'туймазы',
            'zagorjanskas': 'загорянский',
            'novoye devyatkino': 'новое девяткино',
            'yablonovsky': 'яблоновский',
            'petrovo-dalneye': 'петрово-дальнее',
            'kalininets': 'калининец'
        }
        df_copy['geo_city'] = df_copy['geo_city'].replace(cities)

        return df_copy

    def geocode_cities(data):
        import pickle
        from geopy.geocoders import Nominatim

        df_copy = data.copy()

        geolocator = Nominatim(user_agent="geo_app")
        try:
            with open('data/coords_cache.pkl', 'rb') as cache_file:
                coords_cache = pickle.load(cache_file)
        except (FileNotFoundError, EOFError):
            coords_cache = {}

        def get_coordinates(city, coord_type):
            if city not in coords_cache:
                try:
                    if city == '(not set)' or city is None:
                        coords_cache[city] = None
                    else:
                        location = geolocator.geocode(city)
                        if location:
                            coords_cache[city] = {
                                'latitude': location.latitude,
                                'longitude': location.longitude
                            }
                        else:
                            coords_cache[city] = None
                    with open('data/coords_cache.pkl', 'wb') as cache_file:
                        pickle.dump(coords_cache, cache_file)
                except Exception as e:
                    print(f"Error geocoding {city}: {str(e)}")
                    coords_cache[city] = None

            if coords_cache[city]:
                return coords_cache[city][coord_type]
            else:
                return None

        df_copy['latitude'] = df_copy['geo_city'].apply(lambda x: get_coordinates(x, 'latitude'))
        df_copy['longitude'] = df_copy['geo_city'].apply(lambda x: get_coordinates(x, 'longitude'))

        df_copy['latitude'].fillna(df_copy['geo_country'].apply(
            lambda x: get_coordinates(x, 'latitude')), inplace=True)
        df_copy['longitude'].fillna(df_copy['geo_country'].apply(
            lambda x: get_coordinates(x, 'longitude')), inplace=True)

        return df_copy.drop(columns=['geo_city', 'geo_country'])

    def rare_values(data):
        df_copy = data.copy()

        columns_to_filter = ['utm_source', 'utm_medium', 'utm_adcontent', 'utm_campaign',
                             'device_browser', 'device_os', 'device_brand']

        for column in columns_to_filter:
            counts = df_copy[column].value_counts()
            filt = counts[counts < 750].index
            df_copy[column] = df_copy[column].apply(lambda x: 'rare' if x in filt else x)

        return df_copy

    df = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df2 = pd.read_csv('data/ga_hits.csv', low_memory=False)

    df = df_union(df, df2)
    df = df[(df['geo_city'] != '(not set)') | (df['geo_country'] != '(not set)')]

    X = df.drop(['event_action', 'session_id', 'client_id'], axis=1)
    y = df['event_action']


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])

    filter_preprocessor = Pipeline(steps=[
        ('lower_str', FunctionTransformer(lower_str)),
        ('set_device_brand', FunctionTransformer(set_device_brand)),
        ('clean_device_os', FunctionTransformer(clean_device_os)),
        ('fill_other', FunctionTransformer(fill_other)),
        ('set_date', FunctionTransformer(set_date)),
        ('dev_screen', FunctionTransformer(dev_screen)),
        ('device_browser', FunctionTransformer(device_browser)),
        ('is_russia', FunctionTransformer(is_russia)),
        ('is_available', FunctionTransformer(is_available)),
        ('clean_geo_data', FunctionTransformer(clean_geo_data)),
        ('geocode_cities', FunctionTransformer(geocode_cities)),
        ('rare_values', FunctionTransformer(rare_values)),
        ('preprocessor', preprocessor)
    ])

    model = RandomForestClassifier(max_features='sqrt', min_samples_leaf=23, n_estimators=300, bootstrap=False)

    #    models = (
    #    LogisticRegression(solver='liblinear', max_iter=700, penalty='l1'),
    #   RandomForestClassifier(max_features='sqrt', min_samples_leaf=23, n_estimators=300, bootstrap=False),
    #    MLPClassifier()
    # )
    # scores = cross_val_score(pipe, x, y, cv=3, scoring='roc_auc')
    # Model: LogisticRegression, ROC-AUC Mean Score: 0.6817 -
    # Model: RandomForestClassifier, ROC-AUC Mean Score: 0.7068
    # Model: MLPClassifier, ROC-AUC Mean Score: 0.6960

    pipe = Pipeline(steps=[
        ('filter', filter_preprocessor),
        ('classifier', model)
    ])

    pipe.fit(X, y)

    with open('sber_pipe.pkl', 'wb') as file:
        dill.dump(pipe, file)


if __name__ == '__main__':
    main()

