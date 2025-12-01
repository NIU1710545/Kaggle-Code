def comptar_combinacions_equip(df):
    """
    Comptar totes les combinacions de campions per equips (tant t1 com t2) en el DataFrame.
    Retorna un diccionari {combinacio_tuple_sorted: recompte} on la mateixa combinació
    (ordre no importa) s'agrupa independentment de si apareix a l'equip 1 o 2.
    Per cada partida, si ambdós equips tenen exactament la mateixa combinació, 
    només s'incrementa una vegada (no es compta duplicat dins la mateixa fila).
    """
    from collections import Counter
    import pandas as pd

    team1_cols = [f't1_champ{i}id' for i in range(1, 6)]
    team2_cols = [f't2_champ{i}id' for i in range(1, 6)]

    combinacions_counter = Counter()

    def _sorted_tuple_from_cols(row, cols):
        vals = [row[c] for c in cols if pd.notna(row[c])]
        if len(vals) == 0:
            return None
        # assegurem ints i ordre insensible
        try:
            vals_int = tuple(sorted(int(v) for v in vals))
        except Exception:
            return None
        return vals_int

    for _, row in df.iterrows():
        comb1 = _sorted_tuple_from_cols(row, team1_cols)
        comb2 = _sorted_tuple_from_cols(row, team2_cols)

        # evitem comptar dues vegades la mateixa combinació dins la mateixa partida
        combos_a_afegir = {c for c in (comb1, comb2) if c is not None}

        for comb in combos_a_afegir:
            combinacions_counter[comb] += 1

    return dict(combinacions_counter)


def comptar_combinacions_encanteris(df, verbose=False):
    """
    Comptar totes les combinacions d'encanteris per equips (t1 i t2) en el DataFrame.
    Retorna un diccionari {combinacio_tuple_sorted: recompte} on la mateixa combinació
    (ordre no importa) s'agrupa independentment de si apareix a l'equip 1 o 2.
    Per cada partida, cada combinació única (parell ordenat dels dos encanteris d'un campeó)
    es compta una vegada per partida (si el mateix parell apareix en múltiples campeons o equips
    en la mateixa partida, només es compta una vegada).
    """
    from collections import Counter
    import re
    import pandas as pd

    # Encontrar todas las columnas que siguen el patrón de encantamientos
    pattern = re.compile(r'^(t[12])_champ(\d+)_sum([12])$')
    
    # Agrupar las columnas por equipo y campeón
    spells_by_team_champ = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            team, champ_num, spell_num = match.groups()
            key = (team, int(champ_num))
            if key not in spells_by_team_champ:
                spells_by_team_champ[key] = {}
            spells_by_team_champ[key][spell_num] = col
    
    if verbose:
        print(f"Detectades {len(spells_by_team_champ)} campeons amb encanteris.")
        # Mostrar los primeros
        for (team, champ_num), spells in list(spells_by_team_champ.items())[:5]:
            print(f"  {team}, campeó {champ_num}: {spells}")

    # Si no se detectan columnas, retornar vacío
    if not spells_by_team_champ:
        print("No s'han detectat columnes d'encanteris amb el patró esperat.")
        return {}

    combinacions_counter = Counter()

    for idx, row in df.iterrows():
        combinacions_partida = set()
        
        # Para cada equipo y campeón, obtener la combinación de dos encantamientos
        for (team, champ_num), spells in spells_by_team_champ.items():
            # Debería tener dos encantamientos: sum1 y sum2
            if '1' in spells and '2' in spells:
                sum1 = row[spells['1']]
                sum2 = row[spells['2']]
                if pd.notna(sum1) and pd.notna(sum2):
                    try:
                        comb = tuple(sorted((int(sum1), int(sum2))))
                        combinacions_partida.add(comb)
                    except Exception as e:
                        # En caso de error en la conversión a entero, ignorar
                        pass
        
        # Contar cada combinación única de la partida
        for comb in combinacions_partida:
            combinacions_counter[comb] += 1

    if verbose:
        print(f"Total de combinacions úniques trobades: {len(combinacions_counter)}")

    return dict(combinacions_counter)





def afegir_combinacions_equips_a_dataset(df, combinacions_equips_df):
    """
    Afegeix al DataFrame les columnes 'team1_comb_index' i 'team2_comb_index',
    que contenen l'índex de la combinació d'equip corresponent a cada equip
    segons el DataFrame de combinacions proporcionat.
    Si una combinació no es troba, s'assigna NA (pandas Int64 per preservar NA).
    """
    import numpy as np
    import pandas as pd 
    # Creem una clau textual per a cada combinació a combinacions_equips_df (si no existeix)
    key_cols = ['champ1_id','champ2_id','champ3_id','champ4_id','champ5_id']
    if '_key' not in combinacions_equips_df.columns:
        combinacions_equips_df['_key'] = combinacions_equips_df[key_cols].astype(int).astype(str).agg('-'.join, axis=1)

    # Map de clau -> index (ràpid per .map)
    _key_to_index = pd.Series(combinacions_equips_df.index.values, index=combinacions_equips_df['_key']).to_dict()
    # Funció auxiliar per generar sèries de claus a partir de les 5 columnes d'un equip
    def _make_team_keys(df, cols):
        arr = df[cols].values.astype(int)            # (n_rows, 5)
        arr_sorted = np.sort(arr, axis=1)            # ordenar cada fila
        keys = pd.DataFrame(arr_sorted).astype(str).agg('-'.join, axis=1)
        return keys

    team1_cols = [f't1_champ{i}id' for i in range(1, 6)]
    team2_cols = [f't2_champ{i}id' for i in range(1, 6)]

    keys1 = _make_team_keys(df, team1_cols)
    keys2 = _make_team_keys(df, team2_cols)

    # Mappejar les claus als indexos dels combinacions_df; convertir a Int64 per preservar NA
    df['team1_comb_index'] = keys1.map(_key_to_index).astype('Int64')
    df['team2_comb_index'] = keys2.map(_key_to_index).astype('Int64')

    return df  # mantenim nom anterior


def afegir_combinacions_encanteris_a_dataset(df, combinacions_encanteris_df):
    """
    Afegeix al DataFrame columnes per a cada campeó amb l'índex de la combinació d'encanteris.
    Retorna el DataFrame amb les noves columnes i sense les columnes originals d'encanteris.
    """
    import pandas as pd
    import numpy as np
    
    # Crear un diccionari per mapar combinacions d'encanteris a índex
    # Creem una clau textual per a cada combinació
    combinacions_encanteris_df = combinacions_encanteris_df.copy()
    if '_key' not in combinacions_encanteris_df.columns:
        combinacions_encanteris_df['_key'] = combinacions_encanteris_df[['sum1', 'sum2']].astype(str).agg('-'.join, axis=1)
    
    spell_key_to_index = pd.Series(
        combinacions_encanteris_df.index.values, 
        index=combinacions_encanteris_df['_key']
    ).to_dict()
    
    # Funció auxiliar per crear la clau d'encanteris
    def _make_spell_key(sum1, sum2):
        # Ordenem perquè la combinació (3,4) sigui la mateixa que (4,3)
        a, b = int(sum1), int(sum2)
        return f"{min(a, b)}-{max(a, b)}"
    
    # Per a cada campeó de cada equip, crear la nova columna d'índex d'encanteris
    for team in [1, 2]:
        for champ in range(1, 6):
            sum1_col = f't{team}_champ{champ}_sum1'
            sum2_col = f't{team}_champ{champ}_sum2'
            new_col = f't{team}_champ{champ}_spell_index'
            
            # Crear les claus per a cada fila
            spell_keys = df.apply(
                lambda row: _make_spell_key(row[sum1_col], row[sum2_col]), 
                axis=1
            )
            
            # Mapejar a índexos
            df[new_col] = spell_keys.map(spell_key_to_index).astype('Int64')
    
    # Eliminar les columnes originals d'encanteris
    spell_cols_to_drop = []
    for team in [1, 2]:
        for champ in range(1, 6):
            spell_cols_to_drop.extend([f't{team}_champ{champ}_sum1', f't{team}_champ{champ}_sum2'])
    
    df = df.drop(columns=spell_cols_to_drop)
    
    return df


def afegir_totes_combinacions_a_dataset(df, combinacions_equips_df, combinacions_encanteris_df):
    """
    Funció completa que afegeix tant els índexs d'equips com els d'encanteris al dataset.
    """
    # Afegir índexs d'equips (ja tenim la funció)
    df = afegir_combinacions_equips_a_dataset(df, combinacions_equips_df)
    
    # Afegir índexs d'encanteris
    df = afegir_combinacions_encanteris_a_dataset(df, combinacions_encanteris_df)
    
    # Eliminar columnes de campeons individuals (ara tenim els índexs)
    champ_cols_to_drop = []
    for team in [1, 2]:
        for champ in range(1, 6):
            champ_cols_to_drop.append(f't{team}_champ{champ}id')
    
    df = df.drop(columns=champ_cols_to_drop)
    
    return df