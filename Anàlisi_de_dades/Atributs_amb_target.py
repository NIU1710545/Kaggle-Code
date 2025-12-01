
# ------ FUNCIONS GRÀFIQUES --------


def crear_distribucio_target(df, columna, target, bins=None):
    """
    Crea un gràfic de distribució que mostra la relació entre una columna i un target.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd


    # Crear una còpia per no modificar el dataframe original
    df_temp = df.copy()
    
    # Si s'especifiquen bins i la columna és numèrica, crear intervals
    if bins is not None and pd.api.types.is_numeric_dtype(df_temp[columna]):
        df_temp[columna] = pd.cut(df_temp[columna], bins=bins)
    
    # Configurar l'estil de Seaborn
    sns.set(style="whitegrid")
    
    # Crear el gràfic de distribució
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_temp, x=columna, hue=target, palette='Set2')
    
    # Configurar títol i etiquetes
    plt.title(f'Distribució de {target} per {columna}', fontsize=16)
    plt.xlabel(columna, fontsize=14)
    plt.ylabel('Recompte', fontsize=14)
    
    # Mostrar la llegenda
    plt.legend(title=target)
    
    # Ajustar l'espai entre elements
    plt.tight_layout()
    
    # Mostrar el gràfic
    plt.show()



def crear_diagrama_de_sectors(df, columna, target='winner'):
    """
    Crea un diagrama de sectors que mostra la proporció de cada valor únic en una columna.
    """
    import matplotlib.pyplot as plt

    # Comptar els valors únics
    counts = df[columna].value_counts()

    # Crear el diagrama de sectors
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    
    # Configurar títol
    plt.title(f'Diagrama de sectors de {columna}', fontsize=16)
    
    # Mostrar el gràfic
    plt.show()


def analyze_numeric_column(df, col, target='winner', show_plots=True, bins=5, plot_types=None):
    """
    Mostra estadístiques i gràfics per columnes numèriques.
    Paràmetres:
        plot_types: llista amb tipus de gràfic a mostrar. Ex: ['hist', 'count', 'bar']
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if col not in df.columns:
        raise ValueError(f"Columna no trobada: {col}")

    if plot_types is None:
        plot_types = ['hist', 'box']

    if show_plots:
        try:
            sns.set(style="whitegrid")
            if 'hist' in plot_types:
                plt.figure(figsize=(6,3))
                sns.histplot(data=df, x=col, bins=bins, kde=True, color='C0')
                plt.title(f'Distribució de {col}')
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()
            if 'count' in plot_types:
                plt.figure(figsize=(6,3))
                sns.countplot(x=col, data=df, color='C2')
                plt.title(f'Recompte de valors únics per {col}')
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()
            if 'bar' in plot_types and target in df.columns:
                sub = df[[col, target]].dropna()
                # Si bins és None, utilitza els valors únics com a "bins"
                if bins is None:
                    bin_labels = sorted(sub[col].unique())
                    surv_per_bin = sub.groupby(col)[target].sum().reindex(bin_labels)
                    plt.figure(figsize=(7,3))
                    surv_per_bin.plot(kind='bar', color='C1')
                    plt.title(f'Nombre de {target} per valor únic de {col}')
                    plt.ylabel(f'Count de {target}')
                    plt.xlabel(f'{col} (valors únics)')
                    plt.tight_layout()
                    plt.show()
                else:
                    binned = pd.cut(sub[col], bins=bins)
                    surv_per_bin = sub.groupby(binned)[target].sum()
                    plt.figure(figsize=(7,3))
                    surv_per_bin.plot(kind='bar', color='C1')
                    plt.title(f'Nombre de {target} per bins de {col}')
                    plt.ylabel(f'Count de {target}')
                    plt.xlabel(f'{col} (bins)')
                    plt.tight_layout()
                    plt.show()
        except Exception:
            pass




def analitzar_first_event(df, column_name, target='winner', show_percentage=True):
    """
    Analitza la relació entre un 'first' event i el guanyador final.
    Mostra quantes vegades cada equip ha guanyat quan ha aconseguit el 'first'.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset amb les dades
    column_name : str
        Nom de la columna del 'first' event (ex: 'firstBlood', 'firstTower', etc.)
    target : str
        Nom de la columna del guanyador (per defecte: 'winner')
    show_percentage : bool
        Si True, mostra percentatges; si False, mostra counts absoluts
    
    Returns:
    --------
    dict : Diccionari amb les estadístiques de l'análisi
    """

    import matplotlib.pyplot as plt
    import numpy as np
    
    if column_name not in df.columns:
        print(f"ERROR: La columna '{column_name}' no existeix en el DataFrame")
        return None
    
    print("\n" + "="*80)
    print(f"ANÀLISI: {column_name.replace('first', 'FIRST ')}")
    print("="*80)
    print(f"Mostrant: {'PERCENTATGES' if show_percentage else 'COUNTS ABSOLUTS'} de victòries quan es té el 'first'\n")
    
    # Calcular estadístiques per a cada equip
    stats = {}
    
    for team in [1, 2]:
        # Partides on l'equip 'team' va aconseguir el 'first event'
        team_first_events = df[df[column_name] == team]
        team_first_total = len(team_first_events)
        
        # IMPORTANT: Partides on l'equip 'team' va aconseguir el 'first event' I va guanyar
        team_first_and_win = len(team_first_events[team_first_events[target] == team])
        
        # Partides on l'equip 'team' va aconseguir el 'first event' però PERD
        team_first_and_lose = team_first_total - team_first_and_win
        
        if team_first_total > 0:
            win_pct = (team_first_and_win / team_first_total) * 100
            lose_pct = (team_first_and_lose / team_first_total) * 100
            advantage = win_pct - 50
        else:
            win_pct = lose_pct = advantage = 0
        
        stats[team] = {
            'total_first': team_first_total,
            'wins_with_first': team_first_and_win,
            'losses_with_first': team_first_and_lose,
            'win_pct': win_pct,
            'lose_pct': lose_pct,
            'advantage': advantage
        }
        
        # Mostrar estadístiques detallades
        print(f"Equip {team}:")
        print(f"  Total vegades que ha aconseguit el 'first': {team_first_total}")
        print(f"  Vegades que va GUANYAR tenint el 'first': {team_first_and_win} ({win_pct:.2f}%)")
        print(f"  Vegades que va PERDER tenint el 'first': {team_first_and_lose} ({lose_pct:.2f}%)")
        
        if advantage > 5:
            print(f"  ✓ Avantatge SIGNIFICATIU: +{advantage:.2f}%")
        elif advantage > 0:
            print(f"  ✓ Lleu avantatge: +{advantage:.2f}%")
        elif advantage < -5:
            print(f"  ✗ Desavantatge SIGNIFICATIU: {advantage:.2f}%")
        else:
            print(f"  ≈ Impacte mínim: {advantage:.2f}%")
        print()
    
    # Estadístiques globals
    total_predictive_value = abs(stats[1]['advantage']) + abs(stats[2]['advantage'])
    avg_predictive_value = total_predictive_value / 2
    
    total_first_events = stats[1]['total_first'] + stats[2]['total_first']
    total_wins_with_first = stats[1]['wins_with_first'] + stats[2]['wins_with_first']
    
    print("-"*80)
    print("ESTADÍSTIQUES GLOBALS:")
    print("-"*80)
    print(f"Total d'ocasions on algun equip va aconseguir el 'first': {total_first_events}")
    print(f"Total de victòries quan es tenia el 'first': {total_wins_with_first}")
    print(f"Taxa de victòria global si es té el 'first': {(total_wins_with_first/total_first_events*100):.2f}%")
    print(f"Predictibilitat (avantatge mig): {avg_predictive_value:.2f}%")
    
    # Classificació de predictibilitat
    print(f"\n" + "-"*80)
    print("CLASSIFICACIÓ:")
    print("-"*80)
    if avg_predictive_value > 30:
        classification = "🔴 EXTREMADAMENT PREDICTIU"
    elif avg_predictive_value > 15:
        classification = "🟠 MOLT PREDICTIU"
    elif avg_predictive_value > 10:
        classification = "🟡 PREDICTIU"
    elif avg_predictive_value > 5:
        classification = "🟢 MODERADAMENT PREDICTIU"
    elif avg_predictive_value > 2:
        classification = "🔵 DÉBILMENT PREDICTIU"
    else:
        classification = "⚪ NO PREDICTIU"
    
    print(f"{classification}")
    print(f"   Valor: {avg_predictive_value:.2f}%")
    
    # Crear visualització
    print(f"\n" + "-"*80)
    print("VISUALITZACIÓ:")
    print("-"*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dades per als gràfics
    teams = ['Equip 1', 'Equip 2']
    wins = [stats[1]['wins_with_first'], stats[2]['wins_with_first']]
    losses = [stats[1]['losses_with_first'], stats[2]['losses_with_first']]
    x_pos = np.arange(len(teams))
    
    # ========== GRÀFIC 1: COUNTS ABSOLUTS ==========
    ax1.bar(x_pos - 0.2, wins, 0.4, label='Guanya (count)', color='#2ca02c', 
            alpha=0.8, edgecolor='black', linewidth=2)
    ax1.bar(x_pos + 0.2, losses, 0.4, label='Perd (count)', color='#d62728', 
            alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('Nombre de Partides', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Equip', fontsize=11, fontweight='bold')
    ax1.set_title(f'{column_name.replace("first", "FIRST ")} - Victòries quan es té el FIRST (Counts)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(teams)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Afegir labels amb counts
    for i, (win, loss) in enumerate(zip(wins, losses)):
        ax1.text(i - 0.2, win + 50, str(win), ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='#2ca02c')
        ax1.text(i + 0.2, loss + 50, str(loss), ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='#d62728')
    
    # ========== GRÀFIC 2: PERCENTATGES ==========
    win_pcts = [stats[1]['win_pct'], stats[2]['win_pct']]
    lose_pcts = [stats[1]['lose_pct'], stats[2]['lose_pct']]
    
    ax2.bar(x_pos, win_pcts, label='Guanya %', color='#2ca02c', alpha=0.8, 
            edgecolor='black', linewidth=2)
    ax2.bar(x_pos, lose_pcts, bottom=win_pcts, label='Perd %', color='#d62728', 
            alpha=0.8, edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('Percentatge (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Equip', fontsize=11, fontweight='bold')
    ax2.set_title(f'{column_name.replace("first", "FIRST ")} - Taxa de Victòria quan es té el FIRST (Percentatges)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(teams)
    ax2.set_ylim([0, 100])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Afegir labels amb percentatges
    for i, (win_pct, lose_pct) in enumerate(zip(win_pcts, lose_pcts)):
        ax2.text(i, win_pct / 2, f'{win_pct:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=11, color='white')
        ax2.text(i, win_pct + lose_pct / 2, f'{lose_pct:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=11, color='white')
    
    # Afegir línia de referència al 50%
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Equilibri (50%)')
    
    plt.tight_layout()
    plt.show()
    
    # Retornar resultats
    results = {
        'column': column_name,
        'predictibility': avg_predictive_value,
        'classification': classification,
        'stats': stats,
        'total_first_events': total_first_events,
        'total_wins_with_first': total_wins_with_first
    }
    
    print("\n" + "="*80)
    return results

# Provar la funció amb el primer event
print("\nExecutant análisi individual dels 'FIRST' EVENTS...\n")
