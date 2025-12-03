# --- CARREGAR DADES ---
import functools
@functools.lru_cache(maxsize=1)  # "Guarda l'últim resultat"
def carregar_herois_cache():
    import json
    print("🚀 Carregant JSON per PRIMERA vegada...")
    with open("../LOL - Dataset/champion_info_2.json", 'r') as f:
        return json.load(f)


def obtenir_dades_herois(combos, combinacions_equips_df):
    # Carregar els herois com a diccionari (estructura original JSON)
    herois_data = carregar_herois_cache()

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
    herois_data = []

    # Per a cada combo recomanada
    for combo_idx in combos:
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
                
                herois_data.append({
                    'combo_idx': int(combo_idx),
                    'posicio': posicio,
                    'heroi_id': heroi_id,
                    'nom': heroi_info.get('name', 'Desconegut'),
                    'key': heroi_info.get('key', ''),
                    'titol': heroi_info.get('title', ''),
                    'tags': tags_str
                })
            else:
                herois_data.append({
                    'combo_idx': int(combo_idx),
                    'posicio': posicio,
                    'heroi_id': heroi_id,
                    'nom': 'NO TROBAT',
                    'key': '',
                    'titol': '',
                    'tags': ''
                })

    return herois_data



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








def analisi_correlacions_combinacions(DataSet_reduit):
    """
    Analitza com les combinacions d'un equip correlacionen amb les de l'altre.
    """
    import seaborn as sns
    import matplotlib as plt
    import pandas as pd
    
    # Matriu de co-ocurrències
    combos_1 = DataSet_reduit['team1_comb_index'].dropna().astype(int)
    combos_2 = DataSet_reduit['team2_comb_index'].dropna().astype(int)
    
    # Crear matriu de co-ocurrències
    co_occurrence = pd.crosstab(combos_1, combos_2)
    
    # Calcular coeficient de correlació
    correlation = combos_1.corr(combos_2, method='spearman')
    
    # Visualització
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap de co-ocurrències (mostrar només top 20)
    top_20_1 = combos_1.value_counts().head(20).index
    top_20_2 = combos_2.value_counts().head(20).index
    co_occurrence_top = co_occurrence.loc[top_20_1, top_20_2]
    
    sns.heatmap(co_occurrence_top, ax=axes[0], cmap='YlOrRd')
    axes[0].set_title('Co-ocurrències de Combinacions (Top 20)')
    
    # Distribució de diferències
    diferencia = combos_1 - combos_2
    axes[1].hist(diferencia, bins=50, edgecolor='black')
    axes[1].set_title('Distribució de Diferències entre Combinacions')
    axes[1].set_xlabel('Diferència (Equip1 - Equip2)')
    axes[1].set_ylabel('Freqüència')
    
    plt.tight_layout()
    plt.show()
    
    return correlation, co_occurrence




# Per a la secció "Combinació entre els herois de cada equip":

def analisi_profund_combinacions_compartides(DataSet_reduit, combinacions_equips_df, top_n=50):
    """
    Anàlisi més detallat sobre combinacions compartides.
    """
    import pandas as pd


    # 1. Obtenir totes les combinacions guanyadores (no només top 20)
    combos_guanyadors_1 = DataSet_reduit[DataSet_reduit['winner'] == 1]['team1_comb_index'].value_counts()
    combos_guanyadors_2 = DataSet_reduit[DataSet_reduit['winner'] == 2]['team2_comb_index'].value_counts()
    
    # 2. Crear DataFrame comparatiu
    comparativa = pd.DataFrame({
        'combo_idx': list(set(combos_guanyadors_1.index) | set(combos_guanyadors_2.index))
    })
    
    # 3. Afegir estadístiques per cada equip
    stats_per_combo = []
    
    for combo_idx in comparativa['combo_idx']:
        # Stats com a Equip 1
        mascara_team1 = DataSet_reduit['team1_comb_index'] == combo_idx
        partides_team1 = mascara_team1.sum()
        victorias_team1 = (DataSet_reduit.loc[mascara_team1, 'winner'] == 1).sum()
        win_rate_team1 = victorias_team1 / partides_team1 if partides_team1 > 0 else 0
        
        # Stats com a Equip 2
        mascara_team2 = DataSet_reduit['team2_comb_index'] == combo_idx
        partides_team2 = mascara_team2.sum()
        victorias_team2 = (DataSet_reduit.loc[mascara_team2, 'winner'] == 2).sum()
        win_rate_team2 = victorias_team2 / partides_team2 if partides_team2 > 0 else 0
        
        stats_per_combo.append({
            'combo_idx': combo_idx,
            'partides_team1': partides_team1,
            'victorias_team1': victorias_team1,
            'win_rate_team1': win_rate_team1 * 100,
            'partides_team2': partides_team2,
            'victorias_team2': victorias_team2,
            'win_rate_team2': win_rate_team2 * 100,
            'diferencia_win_rate': abs(win_rate_team1 - win_rate_team2) * 100
        })
    
    stats_df = pd.DataFrame(stats_per_combo)
    
    # 4. Filtrar combinacions que hagin guanyat amb ambdós equips (mínim 1 victòria cada)
    combos_amb_exit_ambdos = stats_df[
        (stats_df['victorias_team1'] > 0) & 
        (stats_df['victorias_team2'] > 0)
    ]
    
    # 5. Ordenar per millor rendiment global
    combos_amb_exit_ambdos['win_rate_global'] = (
        combos_amb_exit_ambdos['victorias_team1'] + combos_amb_exit_ambdos['victorias_team2']
    ) / (combos_amb_exit_ambdos['partides_team1'] + combos_amb_exit_ambdos['partides_team2']) * 100
    
    combos_ordenats = combos_amb_exit_ambdos.sort_values('win_rate_global', ascending=False)
    
    print("="*80)
    print("ANÀLISI DE COMBINACIONS COMPARTIDES")
    print("="*80)
    print(f"\nTotal combinacions úniques: {len(stats_df)}")
    print(f"Combinacions amb victòries ambdós equips: {len(combos_amb_exit_ambdos)}")
    
    if len(combos_amb_exit_ambdos) > 0:
        print(f"\nTop {min(10, len(combos_amb_exit_ambdos))} combinacions més versàtils:")
        print("-"*100)
        
        for idx, row in combos_ordenats.head(10).iterrows():
            # Obtenir noms dels herois
            herois = combinacions_equips_df.loc[row['combo_idx']]
            herois_list = herois.tolist() if isinstance(herois, pd.Series) else herois
            
            print(f"\nCombo {int(row['combo_idx'])}:")
            print(f"  Herois IDs: {herois_list}")
            print(f"  Victòries: {int(row['victorias_team1'])} com Team1, {int(row['victorias_team2'])} com Team2")
            print(f"  Win rates: {row['win_rate_team1']:.1f}% (T1) vs {row['win_rate_team2']:.1f}% (T2)")
            print(f"  Win rate global: {row['win_rate_global']:.1f}%")
    else:
        print("\n🚫 CAP combinació ha guanyat partides ambdós equips!")
        print("\nAixò suggereix:")
        print("1. Les combinacions són molt específiques per a cada costat")
        print("2. El sample size pot ser massa petit")
        print("3. Hi ha factors externs (pick order, bans, etc.)")
    
    return combos_ordenats

def obtenir_combinacions_unilaterals(DataSet_reduit, num_equip):
    """
    Troba combinacions que funcionen BÉ per un equip però MAL per l'altre.
    """
    if num_equip == 1:
        col_propi = 'team1_comb_index'
        col_rival = 'team2_comb_index'
    else:
        col_propi = 'team2_comb_index'
        col_rival = 'team1_comb_index'
    
    # Combinacions amb bon win rate per l'equip propi
    combos_propis = []
    for combo in DataSet_reduit[col_propi].dropna().unique():
        mascara = DataSet_reduit[col_propi] == combo
        partides = mascara.sum()
        victorias = (DataSet_reduit.loc[mascara, 'winner'] == num_equip).sum()
        
        if partides >= 10:  # Mínim 10 partides per ser significatiu
            win_rate = victorias / partides
            if win_rate >= 0.6:  # 60%+ win rate
                combos_propis.append((combo, win_rate, partides))
    
    # Ordenar per win rate
    combos_propis.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🚀 Combinacions FORTES per Equip {num_equip} (win rate ≥ 60%):")
    for combo, win_rate, partides in combos_propis[:10]:
        print(f"  Combo {int(combo)}: {win_rate*100:.1f}% ({partides} partides)")
    
    return combos_propis


# --- FUNCIONS ANÀLISI CORRELACIONS COMBINACIONS ---

def crear_matriu_similitud_combinacions(combinacions_equips_df):
    """
    Crea una matriu de similitud entre totes les combinacions d'herois.
    Utilitza la distància de Jaccard per comparar conjunts d'herois.
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import pairwise_distances
    
    # Crear una llista de conjunts d'herois per a cada combinació
    conjunts_herois = []
    for idx, row in combinacions_equips_df.iterrows():
        herois = set([row['champ1_id'], row['champ2_id'], row['champ3_id'], 
                     row['champ4_id'], row['champ5_id']])
        conjunts_herois.append(herois)
    
    # Crear matriu de similitud utilitzant coeficient de Jaccard
    n_combinacions = len(conjunts_herois)
    matriu_similitud = np.zeros((n_combinacions, n_combinacions))
    
    for i in range(n_combinacions):
        for j in range(i, n_combinacions):
            # Coeficient de Jaccard: |A ∩ B| / |A ∪ B|
            interseccio = len(conjunts_herois[i] & conjunts_herois[j])
            unio = len(conjunts_herois[i] | conjunts_herois[j])
            similitud = interseccio / unio if unio > 0 else 0
            
            matriu_similitud[i][j] = similitud
            matriu_similitud[j][i] = similitud  # Matriu simètrica
    
    return matriu_similitud, conjunts_herois


def analitzar_correlacions_performance_optimitzat(DataSet_reduit, combinacions_equips_df, top_n=500):
    """
    Versió optimitzada que analitza només les combinacions més freqüents.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform
    
    print("="*80)
    print("ANÀLISI DE CORRELACIONS (VERSIÓ OPTIMITZADA)")
    print("="*80)
    
    # 1. SELECCIONAR NOMÉS COMBINACIONS FREQÜENTS
    print("\n1. Seleccionant combinacions més freqüents...")
    
    # Contar freqüència de cada combinació
    freq_combo = {}
    for combo_idx in combinacions_equips_df.index:
        mascara_team1 = DataSet_reduit['team1_comb_index'] == combo_idx
        mascara_team2 = DataSet_reduit['team2_comb_index'] == combo_idx
        total_partides = mascara_team1.sum() + mascara_team2.sum()
        if total_partides > 0:
            freq_combo[combo_idx] = total_partides
    
    # Ordenar per freqüència i seleccionar top N
    combos_freq = sorted(freq_combo.items(), key=lambda x: x[1], reverse=True)[:top_n]
    combos_seleccionats = [idx for idx, freq in combos_freq]
    
    print(f"   Total combinacions úniques: {len(combinacions_equips_df)}")
    print(f"   Combinacions seleccionades (top {top_n}): {len(combos_seleccionats)}")
    print(f"   Cobertura aproximada: {sum(freq for _, freq in combos_freq) / len(DataSet_reduit) * 100:.1f}% de partides")
    
    # 2. CALCULAR WIN RATE PER COMBINACIONS SELECCIONADES
    print("\n2. Calculant win rates...")
    
    win_rates = {}
    combo_counts = {}
    
    for combo_idx in combos_seleccionats:
        mascara_team1 = DataSet_reduit['team1_comb_index'] == combo_idx
        mascara_team2 = DataSet_reduit['team2_comb_index'] == combo_idx
        
        victorias_team1 = (DataSet_reduit.loc[mascara_team1, 'winner'] == 1).sum()
        victorias_team2 = (DataSet_reduit.loc[mascara_team2, 'winner'] == 2).sum()
        
        total_partides = mascara_team1.sum() + mascara_team2.sum()
        total_victorias = victorias_team1 + victorias_team2
        
        win_rate = total_victorias / total_partides if total_partides > 0 else 0
        win_rates[combo_idx] = win_rate
        combo_counts[combo_idx] = total_partides
    
    # 3. CALCULAR SIMILITUDS ENTRE COMBINACIONS (MOSTRATGE)
    print("\n3. Calculant similituds (mostratge aleatori)...")
    
    # Crear diccionari de conjunts d'herois per a combinacions seleccionades
    conjunts_herois = {}
    for combo_idx in combos_seleccionats:
        row = combinacions_equips_df.loc[combo_idx]
        herois = set([row['champ1_id'], row['champ2_id'], row['champ3_id'], 
                     row['champ4_id'], row['champ5_id']])
        conjunts_herois[combo_idx] = herois
    
    # Mostrejar parells aleatòriament (no tots contra tots)
    n_parells = min(50000, len(combos_seleccionats) * 10)  # Màxim 50,000 parells
    correlacions = []
    
    np.random.seed(42)  # Per reprodueibilitat
    for _ in range(n_parells):
        i, j = np.random.choice(combos_seleccionats, 2, replace=False)
        
        # Només considerar si ambdues tenen suficients partides
        if combo_counts[i] >= 5 and combo_counts[j] >= 5:
            # Coeficient de Jaccard
            interseccio = len(conjunts_herois[i] & conjunts_herois[j])
            unio = len(conjunts_herois[i] | conjunts_herois[j])
            similitud = interseccio / unio if unio > 0 else 0
            
            dif_win_rate = abs(win_rates[i] - win_rates[j])
            herois_comuns = interseccio
            
            correlacions.append({
                'combo_i': i,
                'combo_j': j,
                'similitud': similitud,
                'diferencia_win_rate': dif_win_rate,
                'herois_comuns': herois_comuns
            })
    
    correl_df = pd.DataFrame(correlacions)
    print(f"   Parells analitzats: {len(correl_df)}")
    
    # 4. VISUALITZACIONS (simplificades)
    if len(correl_df) > 10:
        print("\n4. Generant visualitzacions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gràfica 1: Correlació similitud vs diferencia win rate
        axes[0, 0].scatter(correl_df['similitud'], correl_df['diferencia_win_rate'], 
                          alpha=0.5, s=5, color='blue')
        axes[0, 0].set_xlabel('Similitud (Jaccard)')
        axes[0, 0].set_ylabel('Diferència Win Rate')
        axes[0, 0].set_title(f'Correlació entre {len(correl_df)} parells')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Afegir línia de tendència
        if len(correl_df) > 2:
            z = np.polyfit(correl_df['similitud'], correl_df['diferencia_win_rate'], 1)
            p = np.poly1d(z)
            axes[0, 0].plot(correl_df['similitud'], p(correl_df['similitud']), 
                           "r--", alpha=0.8, label=f"Tendència (r={z[0]:.3f})")
            axes[0, 0].legend()
        
        # Gràfica 2: Histograma de similituds
        axes[0, 1].hist(correl_df['similitud'], bins=20, edgecolor='black', 
                       color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Similitud')
        axes[0, 1].set_ylabel('Freqüència')
        axes[0, 1].set_title('Distribució de Similituds')
        
        # Gràfica 3: Diferencia win rate per herois comuns
        if correl_df['herois_comuns'].nunique() > 1:
            box_data = []
            labels = []
            for k in sorted(correl_df['herois_comuns'].unique()):
                if len(correl_df[correl_df['herois_comuns'] == k]) > 5:
                    box_data.append(correl_df[correl_df['herois_comuns'] == k]['diferencia_win_rate'].values)
                    labels.append(str(k))
            
            if box_data:
                bp = axes[1, 0].boxplot(box_data, labels=labels, patch_artist=True)
                # Colors per a les caixes
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
                for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                    patch.set_facecolor(color)
                axes[1, 0].set_xlabel('Nombre d\'Herois Comuns')
                axes[1, 0].set_ylabel('Diferència Win Rate')
                axes[1, 0].set_title('Diferència de Performance per Herois Comuns')
        
        # Gràfica 4: Heatmap de similitud per a top 15 combinacions
        top_15 = [idx for idx, _ in combos_freq[:15]]
        
        # Crear submatriu petita
        submatriu = np.zeros((15, 15))
        for idx_i, combo_i in enumerate(top_15):
            for idx_j, combo_j in enumerate(top_15):
                if idx_j > idx_i:  # Triangul superior
                    interseccio = len(conjunts_herois[combo_i] & conjunts_herois[combo_j])
                    unio = len(conjunts_herois[combo_i] | conjunts_herois[combo_j])
                    similitud = interseccio / unio if unio > 0 else 0
                    submatriu[idx_i, idx_j] = similitud
                    submatriu[idx_j, idx_i] = similitud
                elif idx_i == idx_j:
                    submatriu[idx_i, idx_j] = 1.0
        
        im = axes[1, 1].imshow(submatriu, cmap='YlOrRd', vmin=0, vmax=1)
        axes[1, 1].set_title('Similitud entre Top 15 Combinacions')
        axes[1, 1].set_xticks(range(15))
        axes[1, 1].set_yticks(range(15))
        axes[1, 1].set_xticklabels([f"C{idx}" for idx in top_15], rotation=45, fontsize=8)
        axes[1, 1].set_yticklabels([f"C{idx}" for idx in top_15], fontsize=8)
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    # 5. ANÀLISI ESTADÍSTIC
    print("\n5. Estadístiques:")
    print("="*40)
    
    if len(correl_df) > 1:
        correlacio_pearson = correl_df['similitud'].corr(correl_df['diferencia_win_rate'])
        print(f"Correlació de Pearson: {correlacio_pearson:.3f}")
        
        if correlacio_pearson < -0.2:
            print("→ Tendència: Com més similars, MÉS S EMBLANÇA el rendiment")
            print("  (Les combinacions similars tendeixen a tenir win rates similars)")
        elif correlacio_pearson > 0.2:
            print("→ Tendència: Com més similars, MÉS DIFERENT el rendiment")
            print("  (Petites diferències importen molt)")
        else:
            print("→ No hi ha correlació clara entre similitud i diferencia de rendiment")
    
    print(f"\nResum de similituds:")
    print(f"  Mitjana: {correl_df['similitud'].mean():.3f}")
    print(f"  Màxim: {correl_df['similitud'].max():.3f}")
    print(f"  Mínim: {correl_df['similitud'].min():.3f}")
    
    # 6. TROBAR EXEMPLES INTERESSANTS
    print("\n6. Exemples destacats:")
    print("="*40)
    
    # Combinacions molt similars (≥ 80%) amb win rates diferents
    similars_alts = correl_df[correl_df['similitud'] >= 0.8].copy()
    if len(similars_alts) > 0:
        similars_alts = similars_alts.sort_values('diferencia_win_rate', ascending=False)
        
        print(f"\nCombinacions MOLT SIMILARS (≥80%) però diferents:")
        for idx, row in similars_alts.head(3).iterrows():
            print(f"\n  Combo {int(row['combo_i'])} ↔ Combo {int(row['combo_j'])}")
            print(f"  Similitud: {row['similitud']:.1%}")
            print(f"  Diferència win rate: {row['diferencia_win_rate']:.1%}")
            print(f"  Herois comuns: {int(row['herois_comuns'])}/5")
            
            # Info adicional sobre els herois diferents
            herois_i = list(conjunts_herois[row['combo_i']])
            herois_j = list(conjunts_herois[row['combo_j']])
            dif_i = [h for h in herois_i if h not in herois_j]
            dif_j = [h for h in herois_j if h not in herois_i]
            
            if dif_i and dif_j:
                print(f"  Heroi diferent Combo {int(row['combo_i'])}: ID {dif_i[0]}")
                print(f"  Heroi diferent Combo {int(row['combo_j'])}: ID {dif_j[0]}")
    
    # Combinacions amb 4 herois comuns
    casi_iguals = correl_df[correl_df['herois_comuns'] == 4].copy()
    if len(casi_iguals) > 0:
        print(f"\nCombinacions amb 4/5 Herois Comuns:")
        for idx, row in casi_iguals.head(2).iterrows():
            print(f"\n  Combo {int(row['combo_i'])} vs Combo {int(row['combo_j'])}")
            print(f"  Diferència win rate: {row['diferencia_win_rate']:.1%}")
    
    return correl_df, win_rates, combos_seleccionats


def analitzar_correlacions_contra_comunes_optimitzat(DataSet_reduit, combinacions_equips_df, min_partides=5):
    """
    Versió optimitzada que funciona amb menys dades.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("="*80)
    print("ANÀLISI DE RENDIMENT CONTRA COMBINACIONS COMUNES")
    print("="*80)
    
    # 1. Identificar combinacions amb suficients partides
    print("\n1. Identificant combinacions amb dades suficients...")
    
    combo_stats = {}
    for combo_idx in combinacions_equips_df.index:
        mascara_team1 = DataSet_reduit['team1_comb_index'] == combo_idx
        mascara_team2 = DataSet_reduit['team2_comb_index'] == combo_idx
        
        partides = mascara_team1.sum() + mascara_team2.sum()
        
        if partides >= min_partides:
            victorias = (DataSet_reduit.loc[mascara_team1, 'winner'] == 1).sum() + \
                       (DataSet_reduit.loc[mascara_team2, 'winner'] == 2).sum()
            win_rate = victorias / partides if partides > 0 else 0
            
            combo_stats[combo_idx] = {
                'partides': partides,
                'win_rate': win_rate
            }
    
    combos_valids = list(combo_stats.keys())
    print(f"   Combinacions amb ≥{min_partides} partides: {len(combos_valids)}")
    
    if len(combos_valids) < 10:
        print("⚠️  POCES DADES: Es necessiten més combinacions per a l'anàlisi")
        return None
    
    # 2. Analitzar enfrontaments entre combinacions
    print("\n2. Analitzant enfrontaments entre combinacions...")
    
    # Crear matriu d'enfrontaments
    enfrontaments = {}
    
    # Només analitzar top combinacions per enfrontaments
    top_combos = sorted(combo_stats.items(), key=lambda x: x[1]['partides'], reverse=True)[:30]
    top_indices = [idx for idx, _ in top_combos]
    
    print(f"   Analitzant enfrontaments entre {len(top_indices)} combinacions comunes...")
    
    for i, combo_i in enumerate(top_indices):
        for combo_j in top_indices[i+1:]:  # Evitar duplicats
            # Buscar partides on s'enfronten
            cond1 = (DataSet_reduit['team1_comb_index'] == combo_i) & \
                   (DataSet_reduit['team2_comb_index'] == combo_j)
            cond2 = (DataSet_reduit['team1_comb_index'] == combo_j) & \
                   (DataSet_reduit['team2_comb_index'] == combo_i)
            
            partides = cond1.sum() + cond2.sum()
            
            if partides >= 2:  # Mínim 2 partides per a alguna significança
                # Victòries del combo_i
                victorias_i = (DataSet_reduit.loc[cond1, 'winner'] == 1).sum() + \
                             (DataSet_reduit.loc[cond2, 'winner'] == 2).sum()
                
                win_rate_i = victorias_i / partides if partides > 0 else 0
                
                enfrontaments[(combo_i, combo_j)] = {
                    'partides': partides,
                    'win_rate_i': win_rate_i,
                    'win_rate_j': 1 - win_rate_i  # Com complementari
                }
    
    print(f"   Enfrontaments amb dades: {len(enfrontaments)}")
    
    if len(enfrontaments) < 5:
        print("⚠️  INSUFICIENTS ENFRONTAMENTS: No hi ha prou partides entre combinacions comunes")
        
        # Mostrar quins enfrontaments sí que tenen dades
        if len(enfrontaments) > 0:
            print("\nEnfrontaments disponibles:")
            for (i, j), stats in list(enfrontaments.items())[:10]:
                print(f"  Combo {i} vs Combo {j}: {stats['partides']} partides")
        
        return None
    
    # 3. Crear matriu de win rates
    print("\n3. Creant matriu de win rates...")
    
    # Crear llista única de totes les combinacions en enfrontaments
    combos_totals = set()
    for i, j in enfrontaments.keys():
        combos_totals.add(i)
        combos_totals.add(j)
    
    combos_list = sorted(list(combos_totals))
    n_combos = len(combos_list)
    
    # Crear matriu
    win_rate_matrix = pd.DataFrame(
        np.full((n_combos, n_combos), np.nan),
        index=combos_list,
        columns=combos_list
    )
    
    # Omplir matriu amb dades
    for (i, j), stats in enfrontaments.items():
        if i in combos_list and j in combos_list:
            idx_i = combos_list.index(i)
            idx_j = combos_list.index(j)
            win_rate_matrix.iloc[idx_i, idx_j] = stats['win_rate_i']
            win_rate_matrix.iloc[idx_j, idx_i] = stats['win_rate_j']
    
    # 4. Anàlisi de consistència
    print("\n4. Analitzant consistència dels 'counters'...")
    
    results = []
    for combo in combos_list:
        # Obtenir tots els win rates d'aquesta combinació contra altres
        win_rates = win_rate_matrix.loc[combo].dropna().values
        
        if len(win_rates) >= 3:  # Mínim contra 3 oponents diferents
            mean_win_rate = np.mean(win_rates)
            std_win_rate = np.std(win_rates)
            n_oponents = len(win_rates)
            
            # Categoria basada en consistència
            if std_win_rate < 0.15:
                categoria = "CONSISTENT"
            elif mean_win_rate > 0.6 and std_win_rate < 0.25:
                categoria = "FORT GENERAL"
            elif mean_win_rate < 0.4 and std_win_rate < 0.25:
                categoria = "DÈBIL GENERAL"
            else:
                categoria = "ESPECIALITZAT"
            
            results.append({
                'combo_idx': combo,
                'oponents': n_oponents,
                'mean_win_rate': mean_win_rate,
                'std_win_rate': std_win_rate,
                'categoria': categoria
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # 5. VISUALITZACIÓ
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Gràfica 1: Distribució de win rates mitjos
        axes[0].hist(results_df['mean_win_rate'], bins=15, edgecolor='black', 
                    color='skyblue', alpha=0.7)
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Equilibri (50%)')
        axes[0].set_xlabel('Win Rate Mitjà')
        axes[0].set_ylabel('Freqüència')
        axes[0].set_title('Distribució de Rendiment Mitjà')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gràfica 2: Win rate vs Consistència (scatter)
        scatter = axes[1].scatter(results_df['mean_win_rate'], results_df['std_win_rate'],
                                 c=results_df['oponents'], cmap='viridis', 
                                 alpha=0.6, s=50)
        axes[1].set_xlabel('Win Rate Mitjà')
        axes[1].set_ylabel('Desviació Estàndard')
        axes[1].set_title('Rendiment vs Consistència')
        axes[1].axhline(y=0.15, color='orange', linestyle=':', label='Llindar Consistència')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Nombre d\'Oponents')
        
        # Gràfica 3: Categories
        cat_counts = results_df['categoria'].value_counts()
        axes[2].bar(cat_counts.index, cat_counts.values, color=['green', 'blue', 'red', 'orange'])
        axes[2].set_xlabel('Categoria')
        axes[2].set_ylabel('Nombre de Combinacions')
        axes[2].set_title('Categorització de Combinacions')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Afegir valors a les barres
        for i, (cat, count) in enumerate(cat_counts.items()):
            axes[2].text(i, count + 0.5, str(count), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # 6. RESULTATS DETALLATS
        print("\n5. Resultats per categories:")
        print("="*50)
        
        for categoria in ['CONSISTENT', 'FORT GENERAL', 'DÈBIL GENERAL', 'ESPECIALITZAT']:
            cat_data = results_df[results_df['categoria'] == categoria]
            if len(cat_data) > 0:
                print(f"\n{categoria} ({len(cat_data)} combinacions):")
                print(f"  Win rate mitjà: {cat_data['mean_win_rate'].mean():.2%}")
                print(f"  Consistència mitjana: {cat_data['std_win_rate'].mean():.2%}")
                
                # Mostrar exemples
                if categoria == "CONSISTENT":
                    print("  Exemples (més consistents):")
                    for _, row in cat_data.sort_values('std_win_rate').head(3).iterrows():
                        print(f"    Combo {int(row['combo_idx'])}: {row['mean_win_rate']:.1%} ± {row['std_win_rate']:.1%}")
                
                elif categoria == "ESPECIALITZAT":
                    print("  Exemples (més especialitzats):")
                    for _, row in cat_data.sort_values('std_win_rate', ascending=False).head(3).iterrows():
                        print(f"    Combo {int(row['combo_idx'])}: {row['mean_win_rate']:.1%} ± {row['std_win_rate']:.1%}")
        
        # 7. TROBAR "COUNTERS" ESPECÍFICS
        print("\n6. Counters específics destacats:")
        print("="*50)
        
        counters = []
        for (i, j), stats in enfrontaments.items():
            if stats['partides'] >= 3:  # Mínim 3 partides
                dif = abs(stats['win_rate_i'] - 0.5)
                if dif > 0.3:  # Més de 80% o menys de 20% win rate
                    counters.append({
                        'combo_i': i,
                        'combo_j': j,
                        'partides': stats['partides'],
                        'win_rate_i': stats['win_rate_i'],
                        'counter_strength': dif
                    })
        
        if counters:
            counters_df = pd.DataFrame(counters)
            counters_df = counters_df.sort_values('counter_strength', ascending=False)
            
            print(f"\nTrobats {len(counters)} counters forts:")
            for idx, row in counters_df.head(5).iterrows():
                if row['win_rate_i'] > 0.7:
                    relacio = f"Combo {int(row['combo_i'])} → Combo {int(row['combo_j'])}"
                    print(f"  {relacio}: {row['win_rate_i']:.0%} win rate ({row['partides']} partides)")
                elif row['win_rate_i'] < 0.3:
                    relacio = f"Combo {int(row['combo_j'])} → Combo {int(row['combo_i'])}"
                    print(f"  {relacio}: {(1-row['win_rate_i']):.0%} win rate ({row['partides']} partides)")
        else:
            print("No s'han trobat counters forts (win rate >70% o <30%)")
    
    return results_df, enfrontaments 



def analitzar_patterns_sinergia(DataSet_reduit, combinacions_equips_df, herois_data):
    """
    Identifica patterns de sinergia entre herois basat en co-ocurrència i win rate.
    """
    import pandas as pd
    import numpy as np
    import itertools
    from collections import defaultdict
    
    # 1. Crear diccionari d'herois per combinació
    combo_herois = {}
    for idx, row in combinacions_equips_df.iterrows():
        herois = [int(row['champ1_id']), int(row['champ2_id']), 
                 int(row['champ3_id']), int(row['champ4_id']), 
                 int(row['champ5_id'])]
        combo_herois[idx] = herois
    
    # 2. Analitzar totes les parelles possibles d'herois
    parelles_stats = defaultdict(list)
    
    for combo_idx, herois in combo_herois.items():
        # Generar totes les parelles úniques d'aquesta combinació
        parelles = list(itertools.combinations(sorted(herois), 2))
        
        # Obtenir win rate d'aquesta combinació
        mascara_team1 = DataSet_reduit['team1_comb_index'] == combo_idx
        mascara_team2 = DataSet_reduit['team2_comb_index'] == combo_idx
        
        victorias = (DataSet_reduit.loc[mascara_team1, 'winner'] == 1).sum() + \
                   (DataSet_reduit.loc[mascara_team2, 'winner'] == 2).sum()
        partides = mascara_team1.sum() + mascara_team2.sum()
        win_rate = victorias / partides if partides > 0 else 0
        
        # Afegir win rate a cada parella d'aquesta combinació
        for parella in parelles:
            parelles_stats[parella].append(win_rate)
    
    # 3. Calcular estadístiques per a cada parella
    results = []
    for parella, win_rates in parelles_stats.items():
        if len(win_rates) >= 3:  # Mínim 3 aparicions
            avg_win_rate = np.mean(win_rates)
            std_win_rate = np.std(win_rates)
            aparicions = len(win_rates)
            
            # Obtenir noms dels herois
            heroi1_nom = herois_data.get(parella[0], {}).get('name', f'ID:{parella[0]}')
            heroi2_nom = herois_data.get(parella[1], {}).get('name', f'ID:{parella[1]}')
            
            results.append({
                'heroi1_id': parella[0],
                'heroi2_id': parella[1],
                'heroi1_nom': heroi1_nom,
                'heroi2_nom': heroi2_nom,
                'aparicions': aparicions,
                'win_rate_promig': avg_win_rate,
                'win_rate_std': std_win_rate,
                'consistencia': 1 - (std_win_rate / avg_win_rate if avg_win_rate > 0 else 1)
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Ordenar per millor sinergia (alt win rate, baixa std)
        results_df['score_sinergia'] = results_df['win_rate_promig'] * (1 - results_df['win_rate_std'])
        results_df = results_df.sort_values('score_sinergia', ascending=False)
        
        print("="*80)
        print("TOP 10 PARELLES AMB MILLOR SINERGIA")
        print("="*80)
        
        for idx, row in results_df.head(10).iterrows():
            print(f"\n{row['heroi1_nom']} + {row['heroi2_nom']}:")
            print(f"  Win rate promig: {row['win_rate_promig']:.2%}")
            print(f"  Consistència: {row['consistencia']:.2%}")
            print(f"  Aparicions: {row['aparicions']} combinacions")
        
        # Identificar parelles amb sinergia forta però poca freqüència (hidden gems)
        hidden_gems = results_df[
            (results_df['win_rate_promig'] > 0.6) & 
            (results_df['aparicions'] <= 5)
        ].copy()
        
        if len(hidden_gems) > 0:
            print("\n\n" + "="*80)
            print("HIDDEN GEMS - Parelles Fortes però Poc Frequents")
            print("="*80)
            
            for idx, row in hidden_gems.head(5).iterrows():
                print(f"\n✨ {row['heroi1_nom']} + {row['heroi2_nom']}:")
                print(f"  Win rate: {row['win_rate_promig']:.2%}")
                print(f"  Només {row['aparicions']} aparicions")
    
    return results_df



# --- FUNCIONS PREPARACIÓ COMBINACIONS HEROS I ENCANTERIS --- 

def comptar_combinacions_equip(df):
    """
    Comptar totes les combinacions de campions per equips (tant t1 com t2) en el DataFrame.
    Retorna un diccionari {combinacio_tuple_sorted: recompte} on la mateixa combinació
    (ordre no importa) s'agrupa independentment de si apareix a l'equip 1 o 2.
    Per cada partida, si ambdós equips tenen exactament la mateixa combinació, 
    només s'incrementa una vegada (no es compta duplicat dins la mateixa fila).
    """
    import numpy as np

    team1_cols = [f't1_champ{i}id' for i in range(1, 6)]
    team2_cols = [f't2_champ{i}id' for i in range(1, 6)]

    # Convertir a numpy array per major velocitat
    team1_data = df[team1_cols].astype(int).to_numpy()
    team2_data = df[team2_cols].astype(int).to_numpy()

    # Ordenar cada fila i convertir a tuples
    team1_sorted = np.sort(team1_data, axis=1)
    team2_sorted = np.sort(team2_data, axis=1)
    
    # Convertir a tuples per poder fer hash
    team1_tuples = [tuple(row) for row in team1_sorted]
    team2_tuples = [tuple(row) for row in team2_sorted]
    
    # Combinar i comptar
    from collections import Counter
    all_combinations = team1_tuples + team2_tuples
    return dict(Counter(all_combinations))


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