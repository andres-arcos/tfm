from librerias import *



def fetch_all_tweets(query, headers, max_total, max_results, carpeta, fecha_dia):
    all_tweets = []
    next_token = None
    url = "https://api.twitter.com/2/tweets/search/recent"
    total = 0
    lista_id_unico = []

    # Calcular rangos de fecha para ese día
    start_time = f"{fecha_dia}T00:00:00Z"
    end_time = f"{fecha_dia}T23:59:59Z"

    while total < max_total:
        params = {
            'query': query,
            'max_results': max_results,
            'tweet.fields': 'id,text,created_at,lang',
            'start_time': start_time,
            'end_time': end_time
        }
        if next_token:
            params['next_token'] = next_token

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f'Error {response.status_code}: {response.text}')
            break

        print("Límite total:", response.headers.get('x-rate-limit-limit'))
        print("Solicitudes restantes:", response.headers.get('x-rate-limit-remaining'))
        print("Se reinicia en:", response.headers.get('x-rate-limit-reset'))

        data = response.json()
        tweets = data.get('data', [])
        lista_id_unico += [i["id"] for i in tweets]
        print("unicos..", len(set(lista_id_unico)))

        for tuit in tweets:
            with open(f"{carpeta}\\{tuit['id']}.json", "w", encoding="utf-8") as f:
                json.dump(tuit, f, ensure_ascii=False, indent=2)

        all_tweets.extend(tweets)
        total += len(tweets)
        print(f'Tweets recolectados: {total}')

        next_token = data.get('meta', {}).get('next_token')
        if not next_token:
            break  # No hay más tweets

        time.sleep(1.5)  # Pausa para no abusar de la API

    return all_tweets, lista_id_unico

def preparar_datos(df, col_texto='texto_limpio', col_label='sent_manual'):
    # Limpieza mínima
    df = df.dropna(subset=[col_texto, col_label]).copy()
    df[col_texto] = df[col_texto].str.lower().str.strip()

    # Codificar etiquetas
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[col_label])

    # Split estratificado
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df[col_texto], y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF estándar para los tres modelos
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        #stop_words='spanish',
        strip_accents='unicode'
    )
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test  = vectorizer.transform(X_test_txt)

    return X_train, X_test, y_train, y_test, vectorizer, encoder

def entrenar_xgboost(X_train, y_train, num_clases):
    modelo = XGBClassifier(
        objective='multi:softprob' if num_clases > 2 else 'binary:logistic',
        num_class=num_clases if num_clases > 2 else None,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        tree_method='hist'  # más rápido
    )
    modelo.fit(X_train, y_train)
    return modelo

def entrenar_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced_subsample'
    )
    rf.fit(X_train, y_train)
    return rf

def evaluar_modelo(modelo, X_test, y_test, encoder):
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(confusion_matrix(y_test, y_pred))

def entrenar_red_neuronal(X_train, y_train, num_clases):
    # Convertir a denso (float32)
    X_train_dense = X_train.astype(np.float32).toarray()

    modelo = Sequential()
    modelo.add(Dense(256, activation='relu', input_shape=(X_train_dense.shape[1],)))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.4))
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.3))
    if num_clases > 2:
        modelo.add(Dense(num_clases, activation='softmax'))
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        modelo.add(Dense(1, activation='sigmoid'))
        modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    modelo.fit(X_train_dense, y_train, batch_size=256, epochs=30, callbacks=[es], verbose=0)
    return modelo

def predecir_red_neuronal(modelo, X_test, num_clases):
    X_test_dense = X_test.astype(np.float32).toarray()
    if num_clases > 2:
        y_prob = modelo.predict(X_test_dense, verbose=0)
        y_pred = y_prob.argmax(axis=1)
    else:
        y_prob = modelo.predict(X_test_dense, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
    return y_pred