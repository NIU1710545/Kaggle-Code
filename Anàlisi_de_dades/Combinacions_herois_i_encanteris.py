

# --- FUNCIONS GRÀFIQUES I ANÀLISI ---

def crear_grafica_i_analisi_de_combinacions_guanyades(DataSet_reduit, col, num_equip, min_win_rate=90):
    import matplotlib.pyplot as plt
    import pandas as pd
    # Analitzar la influència de les combinacions d'herois en les victòries
    print("="*80)
    print("ANÀLISI: INFLUÈNCIA DE LES COMBINACIONS D'HEROIS EN LES VICTÒRIES")
    print("="*80)

    # Obtenir les combinacions més comunes
    top_n = 15
    team_combos = DataSet_reduit[col].value_counts().head(top_n)

    # Per cada combinació, calcular la taxa de victòria quan l'equip 1 la té
    combo_stats = []

    for combo_idx in team_combos.index:
        # Partides on l'equip té aquesta combinació
        matches_with_combo = DataSet_reduit[DataSet_reduit[col] == combo_idx]
        total_matches = len(matches_with_combo)
        
        # Victòries de l'equip amb aquesta combinació
        wins = len(matches_with_combo[matches_with_combo['winner'] == num_equip])
        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
        
        combo_stats.append({
            'combo_idx': combo_idx,
            'total_matches': total_matches,
            'wins': wins,
            'win_rate': win_rate,
            'losses': total_matches - wins
        })

    combo_df = pd.DataFrame(combo_stats)
    combo_df = combo_df.sort_values('win_rate', ascending=False)

    print(f"\nTop {top_n} combinacions d'herois (Equip {num_equip}):")
    print(f"\n{'Combo Idx':<12} {'Partides':<12} {'Victòries':<12} {'Derrotes':<12} {'Taxa Victòria':<15}")
    print("-" * 65)
    for idx, row in combo_df.iterrows():
        print(f"{int(row['combo_idx']):<12} {int(row['total_matches']):<12} {int(row['wins']):<12} {int(row['losses']):<12} {row['win_rate']:>6.2f}%")

    # Crear la gràfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gràfica 1: Taxa de victòria per combinació
    colors = ['#2ca02c' if rate >= 50 else '#d62728' for rate in combo_df['win_rate']]
    ax1.barh(range(len(combo_df)), combo_df['win_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(combo_df)))
    ax1.set_yticklabels([f"Combo {int(idx)}" for idx in combo_df['combo_idx']])
    ax1.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% (equilibri)')
    ax1.axvline(x=min_win_rate, color='orange', linestyle=':', linewidth=2, label=f'{min_win_rate}% (límit recomanat)')
    ax1.set_xlabel('Taxa de Victòria (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Taxa de Victòria per Combinació d\'Herois (Equip {num_equip})', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Afegir percentatges a les barres
    for i, (idx, row) in enumerate(combo_df.iterrows()):
        ax1.text(row['win_rate'] + 1, i, f"{row['win_rate']:.1f}%", va='center', fontsize=9)

    # Gràfica 2: Comparació victòries vs derrotes
    x_pos = range(len(combo_df))
    width = 0.35

    ax2.bar([x - width/2 for x in x_pos], combo_df['wins'], width, label='Victòries', color='#2ca02c', alpha=0.7)
    ax2.bar([x + width/2 for x in x_pos], combo_df['losses'], width, label='Derrotes', color='#d62728', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Combo {int(idx)}" for idx in combo_df['combo_idx']], rotation=45, ha='right')
    ax2.set_ylabel('Nombre de Partides', fontsize=12, fontweight='bold')
    ax2.set_title(f'Victòries vs Derrotes per Combinació (Equip {num_equip})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Estadístiques globals
    print("\n" + "="*80)
    print("ESTADÍSTIQUES GLOBALS:")
    print("="*80)
    taxa_victoria_global = (DataSet_reduit[DataSet_reduit['winner'] == num_equip].shape[0] / len(DataSet_reduit) * 100)
    print(f"Taxa de victòria global de l'equip {num_equip}: {taxa_victoria_global:.2f}%")
    print(f"Combinació amb taxa més alta: Combo {int(combo_df.iloc[0]['combo_idx'])} ({combo_df.iloc[0]['win_rate']:.2f}%)")
    print(f"Combinació amb taxa més baixa: Combo {int(combo_df.iloc[-1]['combo_idx'])} ({combo_df.iloc[-1]['win_rate']:.2f}%)")
    print("="*80)
    
    # Retornar combinacions amb taxa de victòria >= min_win_rate
    combos_recomanats = combo_df[combo_df['win_rate'] >= min_win_rate]['combo_idx'].tolist()
    
    if combos_recomanats:
        print(f"\n✓ COMBINACIONS AMB TAXA >= {min_win_rate}%:")
        print(f"  Índexs: {[int(idx) for idx in combos_recomanats]}")
        print(f"  Total: {len(combos_recomanats)} combinacions")
        for idx in combos_recomanats:
            row = combo_df[combo_df['combo_idx'] == idx].iloc[0]
            print(f"    - Combo {int(idx)}: {row['win_rate']:.2f}% ({int(row['wins'])}/{int(row['total_matches'])} victòries)")
    else:
        print(f"\n✗ No hi ha combinacions amb taxa >= {min_win_rate}%")
    
    return combos_recomanats




def obtenir_dades_herois_recomanats(combos_recomanats, combinacions_equips_df):
    # Carregar els herois com a diccionari (estructura original JSON)
    import json
    with open("../LOL - Dataset/champion_info_2.json", 'r') as f:
        herois_data = json.load(f)

    # Accedir a la secció 'data' que conté tots els herois
    champ_dict = {}
    if 'data' in herois_data:
        for champ_key, champ_info in herois_data['data'].items():
            # La clau pot ser el key (string) o l'id (int), busquem per id
            if isinstance(champ_info, dict) and 'id' in champ_info:
                champ_id = champ_info['id']
                champ_dict[champ_id] = champ_info
            else:
                # Si no té id, saltarem aquesta entrada
                pass

    print(f"Total de herois carregats: {len(champ_dict)}\n")

    # Crear una llista per emmagatzemar les dades
    herois_recomanats_data = []

    # Per a cada combo recomanada
    for combo_idx in combos_recomanats:
        # Obtenir la combinació de herois
        combo_herois = combinacions_equips_df.iloc[int(combo_idx)]
        heroi_ids = [int(combo_herois['champ1_id']), int(combo_herois['champ2_id']), 
                    int(combo_herois['champ3_id']), int(combo_herois['champ4_id']), 
                    int(combo_herois['champ5_id'])]
        
        for posicio, heroi_id in enumerate(heroi_ids, 1):
            if heroi_id in champ_dict:
                heroi_info = champ_dict[heroi_id]
                
                # Processar les tags
                tags = heroi_info.get('tags', [])
                if isinstance(tags, list):
                    tags_str = ', '.join(tags)
                else:
                    tags_str = str(tags)
                
                herois_recomanats_data.append({
                    'combo_idx': int(combo_idx),
                    'posicio': posicio,
                    'heroi_id': heroi_id,
                    'nom': heroi_info.get('name', 'Desconegut'),
                    'key': heroi_info.get('key', ''),
                    'titol': heroi_info.get('title', ''),
                    'tags': tags_str
                })
            else:
                herois_recomanats_data.append({
                    'combo_idx': int(combo_idx),
                    'posicio': posicio,
                    'heroi_id': heroi_id,
                    'nom': 'NO TROBAT',
                    'key': '',
                    'titol': '',
                    'tags': ''
                })

    return herois_recomanats_data

# --- FUNCIONS PREPARACIÓ COMBINACIONS HEROS I ENCANTERIS --- 

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




def reconstruir_campeon(index_triplete, diccionario_tripletes, tripletes_df):
    """Reconstruye la información de un campeón a partir de su índice."""
    # Buscar en el dataframe
    if index_triplete in tripletes_df['index'].values:
        fila = tripletes_df[tripletes_df['index'] == index_triplete].iloc[0]
        return {
            'champ_id': int(fila['champ_id']),
            'sum1': int(fila['sum1']),
            'sum2': int(fila['sum2'])
        }
    return None

"""
# Ejemplo
index_ejemplo = 0
campeon_info = reconstruir_campeon(index_ejemplo, diccionario_tripletes, tripletes_df)
print(f"Índex {index_ejemplo} correspon a: Heroi {campeon_info['champ_id']} amb encanteris ({campeon_info['sum1']}, {campeon_info['sum2']})")

"""