# --- CARREGAR DADES ---
import functools
@functools.lru_cache(maxsize=1)  # "Guarda l'últim resultat"
def carregar_encanteris_cache():
    import json
    print("🚀 Carregant JSON per PRIMERA vegada...")
    with open("../../LOL - Dataset/summoner_spell_info.json", 'r') as f:
        return json.load(f)
    

import functools
@functools.lru_cache(maxsize=1)  # "Guarda l'últim resultat"
def carregar_herois_cache():
    import json
    print("🚀 Carregant JSON per PRIMERA vegada...")
    with open("../../LOL - Dataset/champion_info_2.json", 'r') as f:
        return json.load(f)


# ---- PREPARACIÓ DADES ----

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





# --- Anàlisi de Sinergia entre Encantaments d'un Equip ---


def get_spell_name(spell_id):
    """
    Funció auxiliar per obtenir el nom d'un encanteri a partir del seu ID.
    """
    # Carregar dades dels encanteris
    spell_data = carregar_encanteris_cache()
    spell_id_str = str(spell_id)
    
    if spell_id_str in spell_data:
        return spell_data[spell_id_str]['name']
    return f"Encantament_{spell_id}"

def calcula_sinergia_encanteris(df, min_synergy_games=50):
    from itertools import combinations
    from collections import defaultdict
    import pandas as pd


    """
    Calcula la sinergia entre parells d'encanteris dins d'un mateix equip.
    
    Args:
        df (pd.DataFrame): DataFrame amb les dades de les partides
        min_synergy_games (int): Nombre mínim de partides per considerar una sinergia
    
    Returns:
        dict: Diccionari amb totes les dades de sinergia per a visualització
    """
    # Estructura per emmagatzemar sinergia
    team_spell_synergies_t1 = defaultdict(lambda: {'wins': 0, 'total': 0})
    team_spell_synergies_t2 = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    # Analitzar totes les combinacions de 2 encanteris dins d'un equip
    for idx, row in df.iterrows():
        # EQUIP 1
        spells_t1 = []
        for champ in range(1, 6):
            spell1_col = f't1_champ{champ}_sum1'
            spell2_col = f't1_champ{champ}_sum2'
            if pd.notna(row[spell1_col]):
                spells_t1.append(int(row[spell1_col]))
            if pd.notna(row[spell2_col]):
                spells_t1.append(int(row[spell2_col]))
        
        # Obtenir totes les combinacions úniques de 2 encanteris
        if len(spells_t1) >= 2:
            # Generar combinacions de 2 encanteris
            spell_combinations_t1 = list(combinations(sorted(set(spells_t1)), 2))
            
            for combo in spell_combinations_t1:
                spell1_name = get_spell_name(combo[0])
                spell2_name = get_spell_name(combo[1])
                combo_key = f"{spell1_name} + {spell2_name}"
                
                team_spell_synergies_t1[combo_key]['total'] += 1
                if row['winner'] == 1:
                    team_spell_synergies_t1[combo_key]['wins'] += 1
        
        # EQUIP 2
        spells_t2 = []
        for champ in range(1, 6):
            spell1_col = f't2_champ{champ}_sum1'
            spell2_col = f't2_champ{champ}_sum2'
            if pd.notna(row[spell1_col]):
                spells_t2.append(int(row[spell1_col]))
            if pd.notna(row[spell2_col]):
                spells_t2.append(int(row[spell2_col]))
        
        # Obtenir totes les combinacions úniques de 2 encanteris
        if len(spells_t2) >= 2:
            spell_combinations_t2 = list(combinations(sorted(set(spells_t2)), 2))
            
            for combo in spell_combinations_t2:
                spell1_name = get_spell_name(combo[0])
                spell2_name = get_spell_name(combo[1])
                combo_key = f"{spell1_name} + {spell2_name}"
                
                team_spell_synergies_t2[combo_key]['total'] += 1
                if row['winner'] == 2:
                    team_spell_synergies_t2[combo_key]['wins'] += 1
    
    # Calcular la taxa de victòria esperada (baseline)
    baseline_win_rate_t1 = (df['winner'] == 1).sum() / len(df)
    baseline_win_rate_t2 = (df['winner'] == 2).sum() / len(df)
    
    # Filtrar i ordenar combinacions per equip 1
    sorted_synergies_t1 = sorted(
        [(combo, stats) for combo, stats in team_spell_synergies_t1.items() 
         if stats['total'] >= min_synergy_games],
        key=lambda x: (x[1]['wins'] / x[1]['total'] if x[1]['total'] > 0 else 0),
        reverse=True
    )
    
    # Filtrar i ordenar combinacions per equip 2
    sorted_synergies_t2 = sorted(
        [(combo, stats) for combo, stats in team_spell_synergies_t2.items() 
         if stats['total'] >= min_synergy_games],
        key=lambda x: (x[1]['wins'] / x[1]['total'] if x[1]['total'] > 0 else 0),
        reverse=True
    )
    
    # Preparar dades per a gràfics
    data_synergies_t1 = []
    labels_synergies_t1 = []
    for combo, stats in sorted_synergies_t1[:10]:
        if stats['total'] > 0:
            win_rate = (stats['wins'] / stats['total']) * 100
            synergy = win_rate - (baseline_win_rate_t1 * 100)
            data_synergies_t1.append(synergy)
            labels_synergies_t1.append(combo)
    
    data_synergies_t2 = []
    labels_synergies_t2 = []
    for combo, stats in sorted_synergies_t2[:10]:
        if stats['total'] > 0:
            win_rate = (stats['wins'] / stats['total']) * 100
            synergy = win_rate - (baseline_win_rate_t2 * 100)
            data_synergies_t2.append(synergy)
            labels_synergies_t2.append(combo)
    
    # Calcular estadístiques de sinergia
    synergies_t1_values = []
    for combo, stats in team_spell_synergies_t1.items():
        if stats['total'] >= min_synergy_games:
            win_rate = (stats['wins'] / stats['total']) * 100
            synergy = win_rate - (baseline_win_rate_t1 * 100)
            synergies_t1_values.append(synergy)
    
    synergies_t2_values = []
    for combo, stats in team_spell_synergies_t2.items():
        if stats['total'] >= min_synergy_games:
            win_rate = (stats['wins'] / stats['total']) * 100
            synergy = win_rate - (baseline_win_rate_t2 * 100)
            synergies_t2_values.append(synergy)
    
    # Retornar totes les dades necessàries
    return {
        'baseline_win_rate_t1': baseline_win_rate_t1,
        'baseline_win_rate_t2': baseline_win_rate_t2,
        'team_spell_synergies_t1': dict(team_spell_synergies_t1),
        'team_spell_synergies_t2': dict(team_spell_synergies_t2),
        'sorted_synergies_t1': sorted_synergies_t1,
        'sorted_synergies_t2': sorted_synergies_t2,
        'data_synergies_t1': data_synergies_t1,
        'labels_synergies_t1': labels_synergies_t1,
        'data_synergies_t2': data_synergies_t2,
        'labels_synergies_t2': labels_synergies_t2,
        'synergies_t1_values': synergies_t1_values,
        'synergies_t2_values': synergies_t2_values,
        'min_synergy_games': min_synergy_games
    }






# --- Sinergia ---
def afegir_score_sinergia_encanteris(df, min_synergy_games=50):
    """
    Afegeix columnes de score de sinergia per a cada equip basant-se en les combinacions d'encanteris.
    
    Args:
        df (pd.DataFrame): DataFrame amb les dades de les partides
        min_synergy_games (int): Nombre mínim de partides per considerar una sinergia
    
    Returns:
        pd.DataFrame: DataFrame amb les noves columnes t1_score_sum_sinergia i t2_score_sum_sinergia
    """
    from itertools import combinations
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    
    # Calcular les dades de sinergia
    sinergia_data = calcula_sinergia_encanteris(df, min_synergy_games=min_synergy_games)
    
    # Obtenir el baseline i els diccionaris de sinergia per a cada equip
    baseline_t1 = sinergia_data['baseline_win_rate_t1']
    baseline_t2 = sinergia_data['baseline_win_rate_t2']
    
    # Construir diccionaris de sinergia per a cada equip
    synergy_dict_t1 = {}
    for combo, stats in sinergia_data['team_spell_synergies_t1'].items():
        # Només considerem combinacions amb prou partides (ja filtrades)
        if stats['total'] >= min_synergy_games:
            win_rate = (stats['wins'] / stats['total']) * 100
            synergy = win_rate - (baseline_t1 * 100)
            synergy_dict_t1[combo] = synergy
    
    synergy_dict_t2 = {}
    for combo, stats in sinergia_data['team_spell_synergies_t2'].items():
        if stats['total'] >= min_synergy_games:
            win_rate = (stats['wins'] / stats['total']) * 100
            synergy = win_rate - (baseline_t2 * 100)
            synergy_dict_t2[combo] = synergy
    
    # Funció per obtenir el nom d'un encanteri
    def get_spell_name(spell_id):
        spell_data = carregar_encanteris_cache()
        spell_id_str = str(spell_id)
        if spell_id_str in spell_data:
            return spell_data[spell_id_str]['name']
        return f"Encantament_{spell_id}"
    
    # Inicialitzar llistes per als scores
    scores_t1 = []
    scores_t2 = []
    
    # Per a cada partida, calcular el score de sinergia per a cada equip
    for idx, row in df.iterrows():
        # EQUIP 1
        spells_t1 = []
        for champ in range(1, 6):
            col1 = f't1_champ{champ}_sum1'
            col2 = f't1_champ{champ}_sum2'
            if pd.notna(row[col1]):
                spells_t1.append(int(row[col1]))
            if pd.notna(row[col2]):
                spells_t1.append(int(row[col2]))
        
        # Obtenir combinacions úniques de 2 encanteris dins de l'equip
        unique_spells_t1 = set(spells_t1)
        combos_t1 = list(combinations(unique_spells_t1, 2))
        
        # Calcular score de sinergia per a l'equip 1
        score_t1 = 0
        for combo in combos_t1:
            # Ordenar per assegurar la mateixa clau que al diccionari
            sorted_combo = tuple(sorted(combo))
            # Construir la clau amb els noms dels encanteris
            key = f"{get_spell_name(sorted_combo[0])} + {get_spell_name(sorted_combo[1])}"
            if key in synergy_dict_t1:
                score_t1 += synergy_dict_t1[key]
        
        scores_t1.append(score_t1)
        
        # EQUIP 2
        spells_t2 = []
        for champ in range(1, 6):
            col1 = f't2_champ{champ}_sum1'
            col2 = f't2_champ{champ}_sum2'
            if pd.notna(row[col1]):
                spells_t2.append(int(row[col1]))
            if pd.notna(row[col2]):
                spells_t2.append(int(row[col2]))
        
        # Obtenir combinacions úniques de 2 encanteris dins de l'equip
        unique_spells_t2 = set(spells_t2)
        combos_t2 = list(combinations(unique_spells_t2, 2))
        
        # Calcular score de sinergia per a l'equip 2
        score_t2 = 0
        for combo in combos_t2:
            # Ordenar per assegurar la mateixa clau que al diccionari
            sorted_combo = tuple(sorted(combo))
            # Construir la clau amb els noms dels encanteris
            key = f"{get_spell_name(sorted_combo[0])} + {get_spell_name(sorted_combo[1])}"
            if key in synergy_dict_t2:
                score_t2 += synergy_dict_t2[key]
        
        scores_t2.append(score_t2)
    
    # Afegir les columnes al DataFrame original
    df_copy = df.copy()
    df_copy['t1_score_sum_sinergia'] = scores_t1
    df_copy['t2_score_sum_sinergia'] = scores_t2
    
    return df_copy


def analitzar_correlacio_sinergia_victoria(df):
    """
    Analitza la correlació entre el score de sinergia i la victòria.
    
    Args:
        df (pd.DataFrame): DataFrame amb les columnes de score de sinergia
    
    Returns:
        dict: Diccionari amb estadístiques de correlació
    """
    import numpy as np
    import pandas as pd
    
    # Crear columna de diferència de sinergia entre equips
    df['diferencia_sinergia'] = df['t1_score_sum_sinergia'] - df['t2_score_sum_sinergia']
    
    # Calcular correlació entre diferència de sinergia i victòria
    correlacio = np.corrcoef(df['diferencia_sinergia'], df['winner'])[0, 1]
    
    # Analitzar victòries per rang de diferència de sinergia
    df['rang_diferencia'] = pd.cut(df['diferencia_sinergia'], 
                                   bins=[-float('inf'), -20, -10, 0, 10, 20, float('inf')],
                                   labels=['<-20', '-20 a -10', '-10 a 0', '0 a 10', '10 a 20', '>20'])
    
    win_rates_per_rang = df.groupby('rang_diferencia')['winner'].mean() * 100
    
    # Estadístiques bàsiques
    stats = {
        'correlacio': correlacio,
        'win_rate_t1_positiu': len(df[(df['t1_score_sum_sinergia'] > 0) & (df['winner'] == 1)]) / len(df[df['t1_score_sum_sinergia'] > 0]) * 100,
        'win_rate_t1_negatiu': len(df[(df['t1_score_sum_sinergia'] < 0) & (df['winner'] == 1)]) / len(df[df['t1_score_sum_sinergia'] < 0]) * 100,
        'win_rate_t2_positiu': len(df[(df['t2_score_sum_sinergia'] > 0) & (df['winner'] == 2)]) / len(df[df['t2_score_sum_sinergia'] > 0]) * 100,
        'win_rate_t2_negatiu': len(df[(df['t2_score_sum_sinergia'] < 0) & (df['winner'] == 2)]) / len(df[df['t2_score_sum_sinergia'] < 0]) * 100,
        'win_rates_per_rang': win_rates_per_rang
    }
    
    return stats