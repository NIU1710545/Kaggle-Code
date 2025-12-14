
# --- CARREGAR DADES ---
import functools
@functools.lru_cache(maxsize=1)  # "Guarda l'últim resultat"
def carregar_herois_cache():
    import json
    print("🚀 Carregant JSON per PRIMERA vegada...")
    with open("../../LOL - Dataset/champion_info_2.json", 'r') as f:
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
    if combos_freq:
        total_partides_seleccionades = sum(freq for _, freq in combos_freq)
        cobertura = total_partides_seleccionades / len(DataSet_reduit) * 100
        print(f"   Cobertura aproximada: {cobertura:.1f}% de partides")
    
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
        herois = set([int(row['champ1_id']), int(row['champ2_id']), int(row['champ3_id']), 
                     int(row['champ4_id']), int(row['champ5_id'])])
        conjunts_herois[combo_idx] = herois
    
    # Mostrejar parells aleatòriament (no tots contra tots)
    n_parells = min(50000, len(combos_seleccionats) * 10)  # Màxim 50,000 parells
    correlacions = []
    
    np.random.seed(42)  # Per reprodueibilitat
    
    if len(combos_seleccionats) < 2:
        print("   ERROR: Necessites almenys 2 combinacions seleccionades")
        return pd.DataFrame(), {}, []
    
    for _ in range(n_parells):
        # Seleccionar dos índexs aleatoris diferents
        i_idx, j_idx = np.random.choice(len(combos_seleccionats), 2, replace=False)
        i = combos_seleccionats[i_idx]
        j = combos_seleccionats[j_idx]
        
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
                'similitud': similitud,  # ATENCIÓ: amb accent!
                'diferencia_win_rate': dif_win_rate,
                'herois_comuns': herois_comuns
            })
    
    correl_df = pd.DataFrame(correlacions)
    print(f"   Parells analitzats: {len(correl_df)}")
    
    # 4. VISUALITZACIONS (simplificades)
    if len(correl_df) > 10 and 'similitud' in correl_df.columns:  # VERIFICAR COLUMNA
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
                           "r--", alpha=0.8, label=f"Tendència (pendent={z[0]:.3f})")
            axes[0, 0].legend()
        
        # Gràfica 2: Histograma de similituds
        axes[0, 1].hist(correl_df['similitud'], bins=20, edgecolor='black', 
                       color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Similitud')
        axes[0, 1].set_ylabel('Freqüència')
        axes[0, 1].set_title('Distribució de Similituds')
        
        # Gràfica 3: Diferencia win rate per herois comuns
        if 'herois_comuns' in correl_df.columns and correl_df['herois_comuns'].nunique() > 1:
            box_data = []
            labels = []
            for k in sorted(correl_df['herois_comuns'].unique()):
                data_k = correl_df[correl_df['herois_comuns'] == k]['diferencia_win_rate'].values
                if len(data_k) > 5:
                    box_data.append(data_k)
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
        else:
            axes[1, 0].text(0.5, 0.5, 'No hi ha dades suficients\nper a aquesta gràfica',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Diferència de Performance per Herois Comuns')
        
        # Gràfica 4: Heatmap de similitud per a top 15 combinacions
        top_15_indices = [idx for idx, _ in combos_freq[:15] if idx in conjunts_herois]
        
        if len(top_15_indices) >= 2:
            # Crear submatriu petita
            n_top = len(top_15_indices)
            submatriu = np.zeros((n_top, n_top))
            
            for idx_i, combo_i in enumerate(top_15_indices):
                for idx_j, combo_j in enumerate(top_15_indices):
                    if idx_j > idx_i:  # Triangul superior
                        interseccio = len(conjunts_herois[combo_i] & conjunts_herois[combo_j])
                        unio = len(conjunts_herois[combo_i] | conjunts_herois[combo_j])
                        similitud = interseccio / unio if unio > 0 else 0
                        submatriu[idx_i, idx_j] = similitud
                        submatriu[idx_j, idx_i] = similitud
                    elif idx_i == idx_j:
                        submatriu[idx_i, idx_j] = 1.0
            
            im = axes[1, 1].imshow(submatriu, cmap='YlOrRd', vmin=0, vmax=1)
            axes[1, 1].set_title('Similitud entre Top Combinacions')
            axes[1, 1].set_xticks(range(n_top))
            axes[1, 1].set_yticks(range(n_top))
            axes[1, 1].set_xticklabels([f"C{idx}" for idx in top_15_indices], 
                                      rotation=45, fontsize=8)
            axes[1, 1].set_yticklabels([f"C{idx}" for idx in top_15_indices], 
                                      fontsize=8)
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'No hi ha dades suficients\nper a aquesta gràfica',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Similitud entre Top Combinacions')
        
        plt.tight_layout()
        plt.show()
    
    # 5. ANÀLISI ESTADÍSTIC
    print("\n5. Estadístiques:")
    print("="*40)
    
    if len(correl_df) > 1 and 'similitud' in correl_df.columns:
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
    
    if len(correl_df) > 0 and 'similitud' in correl_df.columns:
        print(f"\nResum de similituds:")
        print(f"  Mitjana: {correl_df['similitud'].mean():.3f}")
        print(f"  Màxim: {correl_df['similitud'].max():.3f}")
        print(f"  Mínim: {correl_df['similitud'].min():.3f}")
    else:
        print("No hi ha dades de similitud disponibles")
    
    # 6. TROBAR EXEMPLES INTERESSANTS
    print("\n6. Exemples destacats:")
    print("="*40)
    
    if len(correl_df) > 0 and 'similitud' in correl_df.columns:
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
                if row['combo_i'] in conjunts_herois and row['combo_j'] in conjunts_herois:
                    herois_i = list(conjunts_herois[row['combo_i']])
                    herois_j = list(conjunts_herois[row['combo_j']])
                    dif_i = [h for h in herois_i if h not in herois_j]
                    dif_j = [h for h in herois_j if h not in herois_i]
                    
                    if dif_i and dif_j:
                        print(f"  Heroi diferent Combo {int(row['combo_i'])}: ID {dif_i[0]}")
                        print(f"  Heroi diferent Combo {int(row['combo_j'])}: ID {dif_j[0]}")
        
        # Combinacions amb 4 herois comuns
        if 'herois_comuns' in correl_df.columns:
            casi_iguals = correl_df[correl_df['herois_comuns'] == 4].copy()
            if len(casi_iguals) > 0:
                print(f"\nCombinacions amb 4/5 Herois Comuns:")
                for idx, row in casi_iguals.head(2).iterrows():
                    print(f"\n  Combo {int(row['combo_i'])} vs Combo {int(row['combo_j'])}")
                    print(f"  Diferència win rate: {row['diferencia_win_rate']:.1%}")
    else:
        print("No hi ha exemples per mostrar")
    
    return correl_df, win_rates, combos_seleccionats


def analitzar_correlacions_contra_comunes_optimitzat(DataSet_reduit, combinacions_equips_df, min_partides=3):
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
    
    if len(combos_valids) < 5:  # Reduït a 5 per ser més permisius
        print(f"⚠️  POCES DADES: Només {len(combos_valids)} combinacions amb ≥{min_partides} partides")
        
        # Retornar dades bàsiques enlloc de None
        results_df = pd.DataFrame([
            {'combo_idx': k, 'partides': v['partides'], 'win_rate': v['win_rate']}
            for k, v in combo_stats.items()
        ])
        enfrontaments = {}
        return results_df, enfrontaments
    
    # 2. Analitzar enfrontaments entre combinacions
    print("\n2. Analitzant enfrontaments entre combinacions...")
    
    # Seleccionar top combinacions per enfrontaments
    top_combos = sorted(combo_stats.items(), key=lambda x: x[1]['partides'], reverse=True)[:20]  # Reduït a 20
    top_indices = [idx for idx, _ in top_combos]
    
    print(f"   Analitzant enfrontaments entre {len(top_indices)} combinacions comunes...")
    
    enfrontaments = {}
    
    for i, combo_i in enumerate(top_indices):
        for combo_j in top_indices[i+1:]:  # Evitar duplicats
            # Buscar partides on s'enfronten
            cond1 = (DataSet_reduit['team1_comb_index'] == combo_i) & \
                   (DataSet_reduit['team2_comb_index'] == combo_j)
            cond2 = (DataSet_reduit['team1_comb_index'] == combo_j) & \
                   (DataSet_reduit['team2_comb_index'] == combo_i)
            
            partides = cond1.sum() + cond2.sum()
            
            if partides >= 2:  # Mínim 2 partides
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
    
    if len(enfrontaments) < 3:
        print(f"⚠️  POCS ENFRONTAMENTS: Només {len(enfrontaments)} enfrontaments amb dades")
        
        # Retornar les dades que sí tenim
        results_df = pd.DataFrame([
            {'combo_idx': k, 'partides': v['partides'], 'win_rate': v['win_rate']}
            for k, v in combo_stats.items()
        ])
        return results_df, enfrontaments
    
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
        
        if len(win_rates) >= 2:  # Reduït a 2 per ser més permisius
            mean_win_rate = np.mean(win_rates)
            std_win_rate = np.std(win_rates) if len(win_rates) > 1 else 0
            n_oponents = len(win_rates)
            
            # Categoria basada en consistència
            if std_win_rate < 0.15:
                categoria = "CONSISTENT"
            elif mean_win_rate > 0.6:
                categoria = "FORT"
            elif mean_win_rate < 0.4:
                categoria = "DÈBIL"
            else:
                categoria = "EQUILIBRAT"
            
            results.append({
                'combo_idx': combo,
                'oponents': n_oponents,
                'mean_win_rate': mean_win_rate,
                'std_win_rate': std_win_rate,
                'categoria': categoria
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # 5. VISUALITZACIÓ (simplificada)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Gràfica 1: Distribució de win rates mitjos
            axes[0].hist(results_df['mean_win_rate'], bins=10, edgecolor='black', 
                        color='skyblue', alpha=0.7)
            axes[0].axvline(x=0.5, color='red', linestyle='--', label='Equilibri (50%)')
            axes[0].set_xlabel('Win Rate Mitjà')
            axes[0].set_ylabel('Freqüència')
            axes[0].set_title('Distribució de Rendiment Mitjà')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Gràfica 2: Categories
            cat_counts = results_df['categoria'].value_counts()
            colors_dict = {
                'CONSISTENT': 'green',
                'FORT': 'blue', 
                'DÈBIL': 'red',
                'EQUILIBRAT': 'orange'
            }
            
            colors = [colors_dict.get(cat, 'gray') for cat in cat_counts.index]
            axes[1].bar(cat_counts.index, cat_counts.values, color=colors)
            axes[1].set_xlabel('Categoria')
            axes[1].set_ylabel('Nombre de Combinacions')
            axes[1].set_title('Categorització de Combinacions')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Afegir valors a les barres
            for i, (cat, count) in enumerate(cat_counts.items()):
                axes[1].text(i, count + 0.1, str(count), ha='center')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"   Error en visualització: {e}")
        
        # 6. RESULTATS DETALLATS
        print("\n5. Resultats per categories:")
        print("="*50)
        
        for categoria in ['CONSISTENT', 'FORT', 'DÈBIL', 'EQUILIBRAT']:
            cat_data = results_df[results_df['categoria'] == categoria]
            if len(cat_data) > 0:
                print(f"\n{categoria} ({len(cat_data)} combinacions):")
                print(f"  Win rate mitjà: {cat_data['mean_win_rate'].mean():.2%}")
                if len(cat_data) > 1:
                    print(f"  Consistència mitjana: {cat_data['std_win_rate'].mean():.2%}")
                
                # Mostrar exemples
                if len(cat_data) > 0:
                    print("  Exemples:")
                    for _, row in cat_data.head(2).iterrows():
                        print(f"    Combo {int(row['combo_idx'])}: {row['mean_win_rate']:.1%} ± {row['std_win_rate']:.1%} "
                              f"(vs {row['oponents']} oponents)")
        
        # 7. TROBAR "COUNTERS" ESPECÍFICS
        print("\n6. Counters específics destacats:")
        print("="*50)
        
        counters = []
        for (i, j), stats in enfrontaments.items():
            if stats['partides'] >= 2:  # Reduït a 2
                if stats['win_rate_i'] > 0.7 or stats['win_rate_i'] < 0.3:
                    counters.append({
                        'combo_i': i,
                        'combo_j': j,
                        'partides': stats['partides'],
                        'win_rate_i': stats['win_rate_i']
                    })
        
        if counters:
            # Ordenar per força del counter
            counters.sort(key=lambda x: abs(x['win_rate_i'] - 0.5), reverse=True)
            
            print(f"\nTrobats {len(counters)} counters:")
            for counter in counters[:5]:  # Mostrar màxim 5
                if counter['win_rate_i'] > 0.7:
                    relacio = f"Combo {int(counter['combo_i'])} → Combo {int(counter['combo_j'])}"
                    print(f"  {relacio}: {counter['win_rate_i']:.0%} ({counter['partides']} partides)")
                elif counter['win_rate_i'] < 0.3:
                    relacio = f"Combo {int(counter['combo_j'])} → Combo {int(counter['combo_i'])}"
                    print(f"  {relacio}: {(1-counter['win_rate_i']):.0%} ({counter['partides']} partides)")
        else:
            print("No s'han trobat counters forts")
    else:
        print("No hi ha resultats per mostrar")
        results_df = pd.DataFrame()
    
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



def analisi_real_diversitat(DataSet_reduit, combinacions_equips_df):
    """
    Anàlisi de la diversitat real del dataset.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("="*80)
    print("ANÀLISI REAL: DIVERSITAT DE COMBINACIONS")
    print("="*80)
    
    # 1. Quantitat real de partides
    n_partides = len(DataSet_reduit)
    print(f"\n1. PARTIDES REALS: {n_partides:,}")
    
    # 2. Comptar aparicions de cada combinació
    aparicions = {}
    for combo_idx in combinacions_equips_df.index:
        aparicions_team1 = (DataSet_reduit['team1_comb_index'] == combo_idx).sum()
        aparicions_team2 = (DataSet_reduit['team2_comb_index'] == combo_idx).sum()
        total_aparicions = aparicions_team1 + aparicions_team2
        if total_aparicions > 0:
            aparicions[combo_idx] = total_aparicions
    
    n_combinacions_amb_dades = len(aparicions)
    print(f"2. COMBINACIONS AMB DADES: {n_combinacions_amb_dades:,}")
    
    # 3. Distribució d'aparicions
    aparicions_series = pd.Series(aparicions)
    
    print(f"\n3. DISTRIBUCIÓ D'APARICIONS:")
    print(f"   - Combinacions que apareixen 1 vegada: {(aparicions_series == 1).sum():,} ({(aparicions_series == 1).sum()/n_combinacions_amb_dades*100:.1f}%)")
    print(f"   - Combinacions que apareixen 2 vegades: {(aparicions_series == 2).sum():,}")
    print(f"   - Combinacions que apareixen 3+ vegades: {(aparicions_series >= 3).sum():,}")
    
    if (aparicions_series >= 3).sum() > 0:
        print(f"   - Màxim d'aparicions: {aparicions_series.max()} vegades")
        print(f"   - Mitjana d'aparicions: {aparicions_series.mean():.2f}")
    
    # 4. GRÀFICA: Distribució de freqüències
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma (log scale per veure millor)
    bins = np.arange(1, aparicions_series.max() + 2) - 0.5
    axes[0].hist(aparicions_series, bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Nombre d\'Aparicions')
    axes[0].set_ylabel('Nombre de Combinacions')
    axes[0].set_title('Distribució de Freqüències de Combinacions')
    axes[0].set_yscale('log')  # Escala logarítmica
    axes[0].grid(True, alpha=0.3)
    
    # Línia acumulativa
    freq_cum = aparicions_series.value_counts().sort_index().cumsum()
    axes[1].plot(freq_cum.index, freq_cum.values, 'b-', marker='o', linewidth=2)
    axes[1].set_xlabel('Mínim d\'Aparicions')
    axes[1].set_ylabel('Combinacions amb ≥ aquestes aparicions')
    axes[1].set_title('Distribució Acumulativa')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Marcar llindars significatius
    for llindar in [2, 3, 5, 10]:
        if llindar in freq_cum.index:
            axes[1].axvline(x=llindar, color='red', linestyle='--', alpha=0.5)
            axes[1].text(llindar, freq_cum[llindar], f' {llindar}', 
                        verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return aparicions_series


def analisi_alternatiu_per_herois(DataSet_reduit, DataSet_herois,combinacions_equips_df, min_partides=10):
    """
    Anàlisi de performance d'herois individuals i parelles bàsiques.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import combinations
    
    print("="*80)
    print("ANÀLISI PER HEROS INDIVIDUALS")
    print("="*80)
    
    # 1. CARREGAR DADES D'HEROIS
    herois_data = carregar_herois_cache()
    
    # Crear diccionari id->nom
    id_to_name = {}
    id_to_tags = {}
    
    if 'data' in herois_data:
        for champ_info in herois_data['data'].values():
            if 'id' in champ_info:
                champ_id = champ_info['id']
                id_to_name[champ_id] = champ_info.get('name', f'ID:{champ_id}')
                id_to_tags[champ_id] = champ_info.get('tags', [])
    
    print(f"\n1. DADES D'HEROIS CARREGADES:")
    print(f"   Herois únics: {len(id_to_name)}")
    
    # 2. CALCULAR WIN RATE PER CADA HEROI
    print("\n2. CALCULANT WIN RATES INDIVIDUALS...")
    
    heroi_stats = {}
    
    for heroi_id in id_to_name.keys():
        # Trobar partides on apareix aquest heroi (a qualsevol equip)
        mascara = pd.Series(False, index=DataSet_reduit.index)
        
        # Comprovar cada posició en cada equip
        for team in [1, 2]:
            for champ in range(1, 6):
                # Necessitem reconstruir quins herois té cada equip
                # Primer, trobar totes les combinacions on apareix aquest heroi
                pass
    
    # Millor enfoc: utilitzar el dataset original sense reduir
    print("\n⚠️  Necessitem el dataset ORIGINAL amb IDs d'herois")
    print("   Utilitzant enfoc alternatiu...")
    
    # 3. ENFOC ALTERNATIU: Utilitzar DataSet_herois (abans de reduir)
    # Assumint que tens aquest dataset disponible
    
    try:
        # Intentem importar el dataset original des del notebook
        # O fer-ho d'una altra manera...
        
        # Crear llista de totes les columnes d'herois
        hero_cols = []
        for team in [1, 2]:
            for champ in range(1, 6):
                hero_cols.append(f't{team}_champ{champ}id')
        
        # Reconstruir dataset temporal amb herois
        temp_df = DataSet_herois[hero_cols + ['winner']].copy()
        
        print(f"\nDataset temporal creat: {len(temp_df)} partides")
        
        # 4. WIN RATE PER HEROI INDIVIDUAL
        print("\n3. ANALITZANT WIN RATE PER HEROI...")
        
        heroi_win_rates = {}
        heroi_pick_rates = {}
        
        total_partides = len(temp_df)
        
        for hero_col in hero_cols:
            # Determinar equip basat en la columna
            team = 1 if 't1_' in hero_col else 2
            
            for idx, row in temp_df.iterrows():
                heroi_id = row[hero_col]
                if pd.notna(heroi_id):
                    heroi_id = int(heroi_id)
                    
                    # Inicialitzar diccionari si no existeix
                    if heroi_id not in heroi_win_rates:
                        heroi_win_rates[heroi_id] = {'wins': 0, 'games': 0}
                    
                    # Comptar partida
                    heroi_win_rates[heroi_id]['games'] += 1
                    
                    # Comptar victòria
                    winner = row['winner']
                    if (team == 1 and winner == 1) or (team == 2 and winner == 2):
                        heroi_win_rates[heroi_id]['wins'] += 1
        
        # Convertir a DataFrame
        heroi_stats_list = []
        for heroi_id, stats in heroi_win_rates.items():
            if stats['games'] >= min_partides:
                win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
                pick_rate = stats['games'] / (total_partides * 10) * 100  # 10 herois per partida
                
                heroi_stats_list.append({
                    'heroi_id': heroi_id,
                    'nom': id_to_name.get(heroi_id, f'ID:{heroi_id}'),
                    'tags': ', '.join(id_to_tags.get(heroi_id, [])),
                    'partides': stats['games'],
                    'victories': stats['wins'],
                    'win_rate': win_rate * 100,
                    'pick_rate': pick_rate
                })
        
        heroi_df = pd.DataFrame(heroi_stats_list)
        
        if len(heroi_df) > 0:
            # Ordenar per win rate
            heroi_df = heroi_df.sort_values('win_rate', ascending=False)
            
            print(f"\n4. RESULTATS (mínim {min_partides} partides):")
            print(f"   Herois analitzats: {len(heroi_df)}")
            
            # Top 10 millors i pitjors
            print("\n   TOP 10 MILLORS HEROS (win rate):")
            for idx, row in heroi_df.head(10).iterrows():
                print(f"     {row['nom']}: {row['win_rate']:.1f}% ({row['partides']} partides)")
            
            print("\n   TOP 10 PITJORS HEROS (win rate):")
            for idx, row in heroi_df.tail(10).iterrows():
                print(f"     {row['nom']}: {row['win_rate']:.1f}% ({row['partides']} partides)")
            
            # 5. VISUALITZACIONS
            print("\n5. GENERANT VISUALITZACIONS...")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gràfica 1: Win rate vs Pick rate
            scatter = axes[0, 0].scatter(heroi_df['pick_rate'], heroi_df['win_rate'], 
                                        c=heroi_df['partides'], cmap='viridis',
                                        alpha=0.6, s=50)
            axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (equilibri)')
            axes[0, 0].set_xlabel('Pick Rate (%)')
            axes[0, 0].set_ylabel('Win Rate (%)')
            axes[0, 0].set_title('Win Rate vs Pick Rate')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0], label='Partides')
            
            # Destacar outliers
            high_win_low_pick = heroi_df[(heroi_df['win_rate'] > 55) & (heroi_df['pick_rate'] < 1)]
            low_win_high_pick = heroi_df[(heroi_df['win_rate'] < 45) & (heroi_df['pick_rate'] > 5)]
            
            for idx, row in high_win_low_pick.iterrows():
                axes[0, 0].annotate(row['nom'], (row['pick_rate'], row['win_rate']), 
                                   fontsize=8, alpha=0.7)
            
            # Gràfica 2: Distribució de win rates
            axes[0, 1].hist(heroi_df['win_rate'], bins=20, edgecolor='black', 
                           color='skyblue', alpha=0.7)
            axes[0, 1].axvline(x=heroi_df['win_rate'].mean(), color='red', 
                              linestyle='--', label=f'Mitjana: {heroi_df["win_rate"].mean():.1f}%')
            axes[0, 1].set_xlabel('Win Rate (%)')
            axes[0, 1].set_ylabel('Nombre de Heros')
            axes[0, 1].set_title('Distribució de Win Rates')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gràfica 3: Win rate per tags (si hi ha dades)
            if 'tags' in heroi_df.columns and heroi_df['tags'].notna().any():
                # Expandir tags
                heroi_expanded = heroi_df.copy()
                heroi_expanded['tags'] = heroi_expanded['tags'].str.split(', ')
                heroi_expanded = heroi_expanded.explode('tags')
                
                # Calcular mitjana per tag
                tag_stats = heroi_expanded.groupby('tags').agg({
                    'win_rate': 'mean',
                    'heroi_id': 'count'
                }).rename(columns={'heroi_id': 'n_herois'})
                
                tag_stats = tag_stats[tag_stats['n_herois'] >= 3]  # Mínim 3 herois per tag
                
                if len(tag_stats) > 0:
                    bars = axes[1, 0].bar(range(len(tag_stats)), tag_stats['win_rate'], 
                                         color='lightgreen', alpha=0.7)
                    axes[1, 0].set_xticks(range(len(tag_stats)))
                    axes[1, 0].set_xticklabels(tag_stats.index, rotation=45, ha='right')
                    axes[1, 0].set_ylabel('Win Rate Mitjà (%)')
                    axes[1, 0].set_title('Win Rate Mitjà per Tag/Rol')
                    axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
                    
                    # Afegir valors
                    for i, (tag, win_rate) in enumerate(zip(tag_stats.index, tag_stats['win_rate'])):
                        axes[1, 0].text(i, win_rate + 0.5, f'{win_rate:.1f}', 
                                       ha='center', fontsize=9)
            
            # Gràfica 4: Partides vs Win rate (consistència)
            axes[1, 1].scatter(heroi_df['partides'], heroi_df['win_rate'], 
                              alpha=0.6, s=30, color='purple')
            axes[1, 1].set_xlabel('Nombre de Partides')
            axes[1, 1].set_ylabel('Win Rate (%)')
            axes[1, 1].set_title('Consistència: Partides vs Win Rate')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Destacar herois consistents (moltes partides, bon win rate)
            consistents = heroi_df[(heroi_df['partides'] > 50) & (heroi_df['win_rate'] > 52)]
            for idx, row in consistents.iterrows():
                axes[1, 1].annotate(row['nom'][:10], (row['partides'], row['win_rate']), 
                                   fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
            # 6. ANÀLISI DE PARELLES BÀSIC
            print("\n6. ANÀLISI BÀSIC DE PARELLES D'HEROS...")
            
            # Analitzar parelles més comunes
            parelles_stats = {}
            
            # Mostrejar partides per eficiència
            sample_size = min(1000, len(temp_df))
            temp_sample = temp_df.sample(sample_size, random_state=42)
            
            for idx, row in temp_sample.iterrows():
                # Obtenir tots els herois d'aquesta partida
                herois_partida = []
                
                # Herois equip 1
                for champ in range(1, 6):
                    hero_id = row[f't1_champ{champ}id']
                    if pd.notna(hero_id):
                        herois_partida.append(int(hero_id))
                
                # Herois equip 2
                for champ in range(1, 6):
                    hero_id = row[f't2_champ{champ}id']
                    if pd.notna(hero_id):
                        herois_partida.append(int(hero_id))
                
                # Generar totes les parelles dins de la partida
                winner = row['winner']
                
                for heroi1, heroi2 in combinations(sorted(herois_partida), 2):
                    parella_key = tuple(sorted([heroi1, heroi2]))
                    
                    if parella_key not in parelles_stats:
                        parelles_stats[parella_key] = {'wins': 0, 'games': 0}
                    
                    parelles_stats[parella_key]['games'] += 1
                    
                    # Determinar si tots dos herois estan al mateix equip guanyador
                    # (Simplificació: assumim que si estan al mateix equip)
                    # En realitat necessitaríem saber quin heroi està a quin equip...
                    # Per ara, saltem aquesta part complexa
            
            print(f"   Parelles analitzades: {len(parelles_stats)}")
            print("   ⚠️  Anàlisi de parelles necessita implementació més avançada")
            
            return heroi_df
            
        else:
            print("⚠️  No hi ha herois amb suficients partides per a l'anàlisi")
            return None
            
    except Exception as e:
        print(f"\n⚠️  ERROR: {e}")
        print("   Assegura't que tens el dataset original 'DataSet_herois' disponible")
        return None



def analisi_per_tags(DataSet_reduit, DataSet_herois, combinacions_equips_df, min_partides):
    """
    Analitza el rendiment basat en composicions de tags/rols en lloc de herois específics.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    import ast
    
    print("="*80)
    print("ANÀLISI PER TAGS/ROLS")
    print("="*80)
    
    # 1. CARREGAR I PREPARAR DADES D'HEROIS
    herois_data = carregar_herois_cache()
    
    # Crear diccionaris
    heroi_to_tags = {}
    all_tags = set()
    
    if 'data' in herois_data:
        for champ_info in herois_data['data'].values():
            if 'id' in champ_info:
                heroi_id = champ_info['id']
                tags = champ_info.get('tags', [])
                if isinstance(tags, list):
                    heroi_to_tags[heroi_id] = tags
                    all_tags.update(tags)
    
    print(f"\n1. TAGS ÚNIQUES IDENTIFICATS: {len(all_tags)}")
    print(f"   Tags: {', '.join(sorted(all_tags))}")
    
    # 2. RECONSTRUIR COMPOSICIONS PER PARTIDA
    print("\n2. ANALITZANT COMPOSICIONS DE TAGS PER PARTIDA...")
    
    composicions_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
    
    # Necessitem el dataset original amb IDs d'herois
    # Assumim que tens DataSet_herois disponible
    
    try:
        # Crear dataset temporal
        hero_cols = []
        for team in [1, 2]:
            for champ in range(1, 6):
                hero_cols.append(f't{team}_champ{champ}id')
        
        temp_df = DataSet_herois[hero_cols + ['winner']].copy()
        
        # Processar cada partida
        for idx, row in temp_df.iterrows():
            winner = row['winner']
            
            # Analitzar cada equip per separat
            for team in [1, 2]:
                tags_equip = []
                
                # Recollir tags de tots els herois de l'equip
                for champ in range(1, 6):
                    hero_id = row[f't{team}_champ{champ}id']
                    if pd.notna(hero_id):
                        hero_id = int(hero_id)
                        if hero_id in heroi_to_tags:
                            tags_equip.extend(heroi_to_tags[hero_id])
                
                # Crear "signatura" de la composició
                # Opció 1: Contar quantitat de cada tag
                tag_counter = Counter(tags_equip)
                
                # Crear clau ordenada (per exemple: "Assassin:1,Fighter:2,Mage:2")
                key_parts = []
                for tag in sorted(all_tags):
                    if tag in tag_counter:
                        key_parts.append(f"{tag}:{tag_counter[tag]}")
                
                composicio_key = ",".join(key_parts) if key_parts else "empty"
                
                # Actualitzar estadístiques
                composicions_stats[composicio_key]['games'] += 1
                
                # Comptar victòria si aquest equip va guanyar
                if (team == 1 and winner == 1) or (team == 2 and winner == 2):
                    composicions_stats[composicio_key]['wins'] += 1
        
        print(f"   Composicions úniques trobades: {len(composicions_stats)}")
        
        # 3. FILTRAR I ORDENAR COMPOSICIONS
        print("\n3. FILTRANT COMPOSICIONS SIGNIFICATIVES...")
        
        composicions_list = []
        for key, stats in composicions_stats.items():
            if stats['games'] >= min_partides:
                win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
                
                # Parsejar la clau per obtenir informació estructurada
                tags_info = {}
                if key != "empty":
                    for part in key.split(','):
                        if ':' in part:
                            tag, count = part.split(':')
                            tags_info[tag] = int(count)
                
                composicions_list.append({
                    'composicio': key,
                    'tags_info': tags_info,
                    'n_tags_diferents': len(tags_info),
                    'total_herois': sum(tags_info.values()) if tags_info else 0,
                    'partides': stats['games'],
                    'victories': stats['wins'],
                    'win_rate': win_rate * 100
                })
        
        composicions_df = pd.DataFrame(composicions_list)
        
        if len(composicions_df) == 0:
            print(f"⚠️  Cap composició amb ≥{min_partides} partides")
            # Provar amb llindar més baix
            min_partides = 2
            composicions_list = []
            for key, stats in composicions_stats.items():
                if stats['games'] >= min_partides:
                    win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
                    tags_info = {}
                    if key != "empty":
                        for part in key.split(','):
                            if ':' in part:
                                tag, count = part.split(':')
                                tags_info[tag] = int(count)
                    
                    composicions_list.append({
                        'composicio': key,
                        'tags_info': tags_info,
                        'partides': stats['games'],
                        'victories': stats['wins'],
                        'win_rate': win_rate * 100
                    })
            
            composicions_df = pd.DataFrame(composicions_list)
        
        if len(composicions_df) > 0:
            # Ordenar per win rate
            composicions_df = composicions_df.sort_values('win_rate', ascending=False)
            
            print(f"\n4. RESULTATS (mínim {min_partides} partides):")
            print(f"   Composicions analitzades: {len(composicions_df)}")
            
            # Mostrar top composicions
            print("\n   TOP 10 MILLORS COMPOSICIONS:")
            for idx, row in composicions_df.head(10).iterrows():
                # Decodificar la composició
                if row['composicio'] == "empty":
                    desc = "Sense tags"
                else:
                    parts = []
                    for tag, count in row['tags_info'].items():
                        parts.append(f"{count}x {tag}")
                    desc = " + ".join(parts)
                
                print(f"     {desc}: {row['win_rate']:.1f}% ({row['partides']} partides)")
            
            # 5. ANÀLISI PER TAG INDIVIDUAL
            print("\n5. ANÀLISI DE TAGS INDIVIDUALS...")
            
            tag_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
            
            for idx, row in temp_df.iterrows():
                winner = row['winner']
                
                for team in [1, 2]:
                    # Recollir tags de l'equip
                    tags_equip = []
                    for champ in range(1, 6):
                        hero_id = row[f't{team}_champ{champ}id']
                        if pd.notna(hero_id):
                            hero_id = int(hero_id)
                            if hero_id in heroi_to_tags:
                                tags_equip.extend(heroi_to_tags[hero_id])
                    
                    # Actualitzar stats per a cada tag
                    equip_guanya = (team == 1 and winner == 1) or (team == 2 and winner == 2)
                    
                    for tag in set(tags_equip):  # Únics per equip
                        tag_stats[tag]['games'] += 1
                        if equip_guanya:
                            tag_stats[tag]['wins'] += 1
            
            # Crear DataFrame de tags
            tag_list = []
            for tag, stats in tag_stats.items():
                if stats['games'] >= 10:  # Mínim 10 aparicions
                    win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
                    tag_list.append({
                        'tag': tag,
                        'partides': stats['games'],
                        'win_rate': win_rate * 100,
                        'pick_rate': (stats['games'] / (len(temp_df) * 2)) * 100  # 2 equips per partida
                    })
            
            tags_df = pd.DataFrame(tag_list)
            
            if len(tags_df) > 0:
                tags_df = tags_df.sort_values('win_rate', ascending=False)
                
                print("\n   WIN RATE PER TAG:")
                for idx, row in tags_df.iterrows():
                    print(f"     {row['tag']}: {row['win_rate']:.1f}% ({row['partides']} aparicions)")
            
            # 6. VISUALITZACIONS
            print("\n6. GENERANT VISUALITZACIONS...")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Gràfica 1: Distribució de win rates de composicions
            if len(composicions_df) > 1:
                axes[0, 0].hist(composicions_df['win_rate'], bins=15, 
                               edgecolor='black', color='lightblue', alpha=0.7)
                axes[0, 0].axvline(x=composicions_df['win_rate'].mean(), 
                                  color='red', linestyle='--',
                                  label=f'Mitjana: {composicions_df["win_rate"].mean():.1f}%')
                axes[0, 0].set_xlabel('Win Rate (%)')
                axes[0, 0].set_ylabel('Nombre de Composicions')
                axes[0, 0].set_title('Distribució Win Rates per Composició')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Gràfica 2: Win rate vs Partides (composicions)
            if len(composicions_df) > 1:
                scatter = axes[0, 1].scatter(composicions_df['partides'], 
                                            composicions_df['win_rate'],
                                            alpha=0.6, s=30, c='green')
                axes[0, 1].set_xlabel('Partides')
                axes[0, 1].set_ylabel('Win Rate (%)')
                axes[0, 1].set_title('Consistència de Composicions')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Destacar composicions interessants
                destacades = composicions_df[
                    (composicions_df['partides'] > composicions_df['partides'].median()) &
                    (abs(composicions_df['win_rate'] - 50) > 10)
                ]
                
                for idx, row in destacades.head(5).iterrows():
                    # Crear descripció curta
                    desc = str(row['tags_info'])[:20] + "..." if row['tags_info'] else "Empty"
                    axes[0, 1].annotate(desc, (row['partides'], row['win_rate']), 
                                       fontsize=7, alpha=0.7)
            
            # Gràfica 3: Barres de win rate per tag
            if len(tags_df) > 0:
                bars = axes[0, 2].bar(range(len(tags_df)), tags_df['win_rate'], 
                                     color='orange', alpha=0.7)
                axes[0, 2].set_xticks(range(len(tags_df)))
                axes[0, 2].set_xticklabels(tags_df['tag'], rotation=45, ha='right')
                axes[0, 2].set_ylabel('Win Rate (%)')
                axes[0, 2].set_title('Win Rate per Tag Individual')
                axes[0, 2].axhline(y=50, color='red', linestyle='--', alpha=0.5)
                axes[0, 2].grid(True, alpha=0.3, axis='y')
                
                # Afegir valors
                for i, win_rate in enumerate(tags_df['win_rate']):
                    axes[0, 2].text(i, win_rate + 0.5, f'{win_rate:.1f}', 
                                   ha='center', fontsize=9)
            
            # Gràfica 4: Heatmap de co-ocurrència de tags (simplificat)
            if len(composicions_df) > 5 and 'tags_info' in composicions_df.columns:
                # Crear llista de totes les parelles de tags
                tag_pairs = defaultdict(int)
                
                for idx, row in composicions_df.iterrows():
                    tags = list(row['tags_info'].keys())
                    # Generar totes les parelles úniques
                    for i in range(len(tags)):
                        for j in range(i+1, len(tags)):
                            pair = tuple(sorted([tags[i], tags[j]]))
                            tag_pairs[pair] += row['partides']
                
                # Convertir a DataFrame per a visualització
                if tag_pairs:
                    pairs_df = pd.DataFrame([
                        {'tag1': p[0], 'tag2': p[1], 'freq': freq}
                        for p, freq in tag_pairs.items()
                    ])
                    
                    # Pivotar per crear matriu
                    pivot_df = pairs_df.pivot(index='tag1', columns='tag2', values='freq')
                    
                    # Plot
                    if not pivot_df.empty:
                        im = axes[1, 0].imshow(pivot_df.fillna(0), cmap='YlOrRd')
                        axes[1, 0].set_title('Co-ocurrència de Tags')
                        axes[1, 0].set_xlabel('Tag 2')
                        axes[1, 0].set_ylabel('Tag 1')
                        plt.colorbar(im, ax=axes[1, 0])
            
            # Gràfica 5: Número de tags diferents vs Win rate
            if 'n_tags_diferents' in composicions_df.columns:
                axes[1, 1].scatter(composicions_df['n_tags_diferents'], 
                                  composicions_df['win_rate'],
                                  alpha=0.6, s=30, c='purple')
                axes[1, 1].set_xlabel('Nombre de Tags Diferents')
                axes[1, 1].set_ylabel('Win Rate (%)')
                axes[1, 1].set_title('Diversitat de Tags vs Rendiment')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Calcular mitjana per cada número de tags
                for n in sorted(composicions_df['n_tags_diferents'].unique()):
                    subset = composicions_df[composicions_df['n_tags_diferents'] == n]
                    if len(subset) > 0:
                        mean_win = subset['win_rate'].mean()
                        axes[1, 1].axhline(y=mean_win, xmin=(n-0.5)/5, xmax=(n+0.5)/5, 
                                          color='red', alpha=0.5, linewidth=2)
            
            # Gràfica 6: Tags més comuns en composicions guanyadores
            if len(composicions_df) > 0:
                # Analitzar tags de composicions amb win rate > 55%
                winning_comps = composicions_df[composicions_df['win_rate'] > 55]
                
                if len(winning_comps) > 0:
                    winning_tags = defaultdict(int)
                    
                    for idx, row in winning_comps.iterrows():
                        for tag, count in row['tags_info'].items():
                            winning_tags[tag] += count * row['partides']
                    
                    # Ordenar
                    sorted_tags = sorted(winning_tags.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    if sorted_tags:
                        tags_names = [t[0] for t in sorted_tags]
                        tags_counts = [t[1] for t in sorted_tags]
                        
                        bars = axes[1, 2].bar(range(len(tags_names)), tags_counts, 
                                             color='lightgreen', alpha=0.7)
                        axes[1, 2].set_xticks(range(len(tags_names)))
                        axes[1, 2].set_xticklabels(tags_names, rotation=45, ha='right')
                        axes[1, 2].set_ylabel('Freqüència (ponderada)')
                        axes[1, 2].set_title('Tags més comuns en composicions guanyadores (>55% WR)')
                        
                        # Afegir valors
                        for i, count in enumerate(tags_counts):
                            axes[1, 2].text(i, count + max(tags_counts)*0.01, 
                                           f'{count}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            # 7. CONCLUSIONS
            print("\n7. CONCLUSIONS CLAU:")
            print("="*50)
            
            if len(tags_df) > 0:
                best_tag = tags_df.iloc[0]
                worst_tag = tags_df.iloc[-1]
                print(f"   Millor tag: {best_tag['tag']} ({best_tag['win_rate']:.1f}% win rate)")
                print(f"   Pitjor tag: {worst_tag['tag']} ({worst_tag['win_rate']:.1f}% win rate)")
            
            if len(composicions_df) > 0:
                best_comp = composicions_df.iloc[0]
                if best_comp['tags_info']:
                    desc = " + ".join([f"{count}x {tag}" for tag, count in best_comp['tags_info'].items()])
                    print(f"   Millor composició: {desc}")
                    print(f"   Win rate: {best_comp['win_rate']:.1f}% ({best_comp['partides']} partides)")
            
            # Recomanació general
            print(f"\n   Recomanació: {'Utilitzar anàlisi per tags' if len(composicions_df) > 20 else 'Diversitat massa alta per conclusions fiables'}")
            
            return composicions_df, tags_df
            
        else:
            print("⚠️  No hi ha dades suficients per a l'anàlisi")
            return None, None
            
    except Exception as e:
        print(f"\n⚠️  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

















# --- FUNCIONS PREPARACIÓ COMBINACIONS HEROS I ENCANTERIS --- 

def comptar_combinacions_equip(df):
    """
    Comptar totes les combinacions de campions per equips (tant t1 com t2) en el DataFrame.
    Retorna un diccionari {combinacio_tuple_sorted: recompte} on la mateixa combinació
    (ordre no importa) s'agrupa independentment de si apareix a l'equip 1 o 2.
    Per cada partida, si ambdós equips tenen exactament la mateixa combinació, 
    només s'incrementa una vegada (no es compta duplicat dins la mateixa fila).
    **IMPORTANT**: Una combinació només es compta UNA vegada total, encara que aparegui
    en múltiples partides o en equips diferents.
    """
    import numpy as np
    from collections import defaultdict
    
    team1_cols = [f't1_champ{i}id' for i in range(1, 6)]
    team2_cols = [f't2_champ{i}id' for i in range(1, 6)]
    
    # Convertir a numpy array per major velocitat
    team1_data = df[team1_cols].astype(int).to_numpy()
    team2_data = df[team2_cols].astype(int).to_numpy()
    
    # Ordenar cada fila i convertir a tuples
    team1_sorted = np.sort(team1_data, axis=1)
    team2_sorted = np.sort(team2_data, axis=1)
    
    # Utilitzem un diccionari per registrar les combinacions úniques
    # Si una combinació ja s'ha vist abans, no la tornem a comptar
    combinacions_vistes = set()
    recomptes = defaultdict(int)
    
    # Processar cada partida (fila)
    for i in range(len(df)):
        t1_tuple = tuple(team1_sorted[i])
        t2_tuple = tuple(team2_sorted[i])
        
        # Si els dos equips tenen la mateixa combinació, només processem una vegada
        if t1_tuple == t2_tuple:
            if t1_tuple not in combinacions_vistes:
                combinacions_vistes.add(t1_tuple)
                recomptes[t1_tuple] += 1
        else:
            # Processar equip 1
            if t1_tuple not in combinacions_vistes:
                combinacions_vistes.add(t1_tuple)
                recomptes[t1_tuple] += 1
            
            # Processar equip 2
            if t2_tuple not in combinacions_vistes:
                combinacions_vistes.add(t2_tuple)
                recomptes[t2_tuple] += 1
    
    return dict(recomptes)



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
