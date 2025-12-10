

def carregar_herois_cache():
    import json
    print("🚀 Carregant JSON per PRIMERA vegada...")
    with open("../../LOL - Dataset/champion_info_2.json", 'r') as f:
        return json.load(f)


def calcular_sinergia_equips(DataSet_reduit, combinacions_equips_df, sinergia_df):
    """
    Calcula el score de sinergia promig per a cada equip i l'afegeix a la base de dades.
    
    Args:
        DataSet_reduit: DataFrame amb les dades de partides
        combinacions_equips_df: DataFrame amb les combinacions d'equips
        sinergia_df: DataFrame amb les sinergies de parelles (generat per analitzar_patterns_sinergia)
    
    Returns:
        DataFrame amb les noves columnes de sinergia afegides
    """
    import pandas as pd
    import numpy as np
    import itertools
    
    # Crear un diccionari ràpid per a cercar sinergies de parelles
    sinergia_dict = {}
    for _, row in sinergia_df.iterrows():
        # Crear una clau única ordenada per a la parella
        key = tuple(sorted([int(row['heroi1_id']), int(row['heroi2_id'])]))
        sinergia_dict[key] = row['score_sinergia']
    
    # Funció per obtenir herois d'una combinació
    def obtenir_herois_combinacio(comb_index):
        if comb_index in combinacions_equips_df.index:
            row = combinacions_equips_df.loc[comb_index]
            return [
                int(row['champ1_id']),
                int(row['champ2_id']),
                int(row['champ3_id']),
                int(row['champ4_id']),
                int(row['champ5_id'])
            ]
        return []
    
    # Funció per calcular sinergia d'un equip
    def calcular_sinergia_promig_equip(herois):
        if len(herois) != 5:
            return 0.5  # Valor per defecte si no hi ha 5 herois
        
        # Generar totes les parelles possibles (10 parelles en total)
        parelles = list(itertools.combinations(sorted(herois), 2))
        
        # Obtenir scores de sinergia per a cada parella
        scores = []
        for parella in parelles:
            key = tuple(sorted(parella))
            score = sinergia_dict.get(key, 0.5)  # 0.5 per defecte si no hi ha dades
            scores.append(score)
        
        # Calcular promig, normalitzat perquè no sigui 1.0 perfecte
        if scores:
            sinergia_promig = np.mean(scores)
            # Normalitzem lleugerament per evitar valors extrems
            sinergia_promig = 0.5 + (sinergia_promig - 0.5) * 0.9
            return np.minimum(sinergia_promig, 0.95)  # CORREGIT: np.minimum en lloc de min
        return 0.5
    
    # Calcular sinergia per a cada equip en cada partida
    team1_sinergies = []
    team2_sinergies = []
    
    for idx, row in DataSet_reduit.iterrows():
        # Obtenir herois de cada equip
        herois_team1 = obtenir_herois_combinacio(row['team1_comb_index'])
        herois_team2 = obtenir_herois_combinacio(row['team2_comb_index'])
        
        # Calcular sinergia promig
        sinergia_team1 = calcular_sinergia_promig_equip(herois_team1)
        sinergia_team2 = calcular_sinergia_promig_equip(herois_team2)
        
        team1_sinergies.append(sinergia_team1)
        team2_sinergies.append(sinergia_team2)
    
    # Afegir columnes a la base de dades
    DataSet_reduit['team1_sinergia_promig'] = team1_sinergies
    DataSet_reduit['team2_sinergia_promig'] = team2_sinergies
    
    # Calcular diferència de sinergia (pot ser útil per al model)
    DataSet_reduit['sinergia_diferencia'] = DataSet_reduit['team1_sinergia_promig'] - DataSet_reduit['team2_sinergia_promig']
    
    print("="*60)
    print("ESTADÍSTIQUES DE SINERGIA AFEGIDES")
    print("="*60)
    print(f"Team1 Sinergia Promig: {DataSet_reduit['team1_sinergia_promig'].mean():.3f}")
    print(f"Team2 Sinergia Promig: {DataSet_reduit['team2_sinergia_promig'].mean():.3f}")
    print(f"Diferència Mitjana: {DataSet_reduit['sinergia_diferencia'].mean():.3f}")
    
    return DataSet_reduit


def analitzar_patterns_sinergia(DataSet_reduit, combinacions_equips_df, herois_data):
    """
    Identifica patterns de sinergia entre herois basat en co-ocurrència i win rate.
    Modificada per integrar-se amb el flux de modificació de la base de dades.
    """
    import pandas as pd
    import numpy as np
    import itertools
    from collections import defaultdict
    
    print("="*60)
    print("ANÀLISI DE SINERGIA ENTRE HEROIS")
    print("="*60)
    
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
        win_rate = victorias / partides if partides > 0 else 0.5
        
        # Afegir win rate a cada parella d'aquesta combinació
        for parella in parelles:
            parelles_stats[parella].append(win_rate)
    
    # 3. Calcular estadístiques per a cada parella
    results = []
    for parella, win_rates in parelles_stats.items():
        if len(win_rates) >= 2:  # Reduït a mínim 2 aparacions per tenir més dades
            avg_win_rate = np.mean(win_rates)
            std_win_rate = np.std(win_rates) if len(win_rates) > 1 else 0.1
            aparicions = len(win_rates)
            
            # Obtenir noms dels herois
            heroi1_nom = herois_data.get(parella[0], {}).get('name', f'ID:{parella[0]}')
            heroi2_nom = herois_data.get(parella[1], {}).get('name', f'ID:{parella[1]}')
            
            # Calcular consistència (inversa de la variabilitat)
            consistencia = 1 - (std_win_rate / np.maximum(avg_win_rate, 0.1))
            consistencia = np.clip(consistencia, 0, 1)  # Normalitzar entre 0 i 1
            
            # Calcular score de sinergia (més equilibrat)
            # Combina win rate i consistència, amb pes per aparicions
            pes_aparicions = np.minimum(aparicions / 10, 1)  # CORREGIT: np.minimum en lloc de min
            score_sinergia = (avg_win_rate * 0.6 + consistencia * 0.4) * pes_aparicions
            
            results.append({
                'heroi1_id': parella[0],
                'heroi2_id': parella[1],
                'heroi1_nom': heroi1_nom,
                'heroi2_nom': heroi2_nom,
                'aparicions': aparicions,
                'win_rate_promig': avg_win_rate,
                'win_rate_std': std_win_rate,
                'consistencia': consistencia,
                'score_sinergia': score_sinergia
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Ordenar per millor sinergia
        results_df = results_df.sort_values('score_sinergia', ascending=False).reset_index(drop=True)
        
        print(f"\nParelles analitzades: {len(results_df)}")
        print("\nTOP 10 PARELLES AMB MILLOR SINERGIA:")
        print("-" * 50)
        
        for idx, row in results_df.head(10).iterrows():
            print(f"{row['heroi1_nom']:15} + {row['heroi2_nom']:15} | "
                  f"WR: {row['win_rate_promig']:.2%} | "
                  f"Sinergia: {row['score_sinergia']:.3f} | "
                  f"Aparicions: {row['aparicions']}")
        
        # Estadístiques generals
        print("\n" + "="*50)
        print("ESTADÍSTIQUES GENERALS DE SINERGIA:")
        print("="*50)
        print(f"Score sinergia mitjà: {results_df['score_sinergia'].mean():.3f}")
        print(f"Score sinergia màxim: {results_df['score_sinergia'].max():.3f}")
        print(f"Score sinergia mínim: {results_df['score_sinergia'].min():.3f}")
        
        # Histograma de scores
        print("\nDistribució de scores de sinergia:")
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(results_df['score_sinergia'], bins=bins)
        for i in range(len(bins)-1):
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} parelles")
    
    return results_df


# FLUX COMPLET D'IMPLEMENTACIÓ
def implementar_sinergia_model(DataSet_reduit, combinacions_equips_df):
    """
    Flux complet per implementar la sinergia al model.
    
    1. Analitza les sinergies entre parelles
    2. Calcula el score de sinergia promig per equip
    3. Afegeix les columnes a la base de dades original
    """
    # 1. Carregar dades dels herois
    herois_json = carregar_herois_cache()
    herois_data = {}
    if 'data' in herois_json:
        for champ_info in herois_json['data'].values():
            if 'id' in champ_info:
                herois_data[champ_info['id']] = champ_info
    
    # 2. Analitzar sinergies entre parelles
    sinergia_df = analitzar_patterns_sinergia(DataSet_reduit, combinacions_equips_df, herois_data)
    
    # 3. Calcular sinergia per equips i afegir a la base de dades
    DataSet_amb_sinergia = calcular_sinergia_equips(DataSet_reduit.copy(), combinacions_equips_df, sinergia_df)
    
    print("\n" + "="*60)
    print("IMPLEMENTACIÓ COMPLETADA")
    print("="*60)
    print(f"Columnes afegides: team1_sinergia_promig, team2_sinergia_promig, sinergia_diferencia")
    print(f"Total de files processades: {len(DataSet_amb_sinergia)}")
    
    return DataSet_amb_sinergia, sinergia_df


def analisi_per_tags(DataSet_reduit, DataSet_herois, combinacions_equips_df, min_partides=5):
    """
    Analitza el rendiment basat en composicions de tags/rols i afegeix score de tags a DataSet_reduit.
    Retorna DataSet_reduit amb noves columnes: team1_tag_score, team2_tag_score, tag_score_diff.
    """
    import pandas as pd
    import numpy as np
    from collections import Counter, defaultdict
    
    print("="*80)
    print("ANÀLISI PER TAGS/ROLS I CÀLCUL DE SCORE")
    print("="*80)
    
    # 1. CARREGAR I PREPARAR DADES D'HEROIS
    herois_data = carregar_herois_cache()
    
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
    
    # 2. PREPROCESSAR SIGNATURES DE TAGS PER COMBINACIONS D'EQUIP
    print("\n2. PREPROCESSANT SIGNATURES DE TAGS PER COMBINACIONS D'EQUIP...")
    
    combo_to_signatura = {}
    signatura_to_combos = defaultdict(list)
    
    for combo_idx, row in combinacions_equips_df.iterrows():
        herois = [int(row['champ1_id']), int(row['champ2_id']), 
                  int(row['champ3_id']), int(row['champ4_id']), 
                  int(row['champ5_id'])]
        
        tags_equip = []
        for hero_id in herois:
            if hero_id in heroi_to_tags:
                tags_equip.extend(heroi_to_tags[hero_id])
        
        tag_counter = Counter(tags_equip)
        key_parts = []
        for tag in sorted(all_tags):
            if tag in tag_counter:
                key_parts.append(f"{tag}:{tag_counter[tag]}")
        
        signatura = ",".join(key_parts) if key_parts else "empty"
        combo_to_signatura[combo_idx] = signatura
        signatura_to_combos[signatura].append(combo_idx)
    
    print(f"   Combinacions processades: {len(combo_to_signatura)}")
    print(f"   Signatures úniques: {len(signatura_to_combos)}")
    
    # 3. CALCULAR ESTADÍSTIQUES PER SIGNATURA
    print("\n3. CALCULANT ESTADÍSTIQUES PER SIGNATURA...")
    
    signatura_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
    total_partides = len(DataSet_herois)
    
    for idx, row in DataSet_herois.iterrows():
        winner = row['winner']
        
        for team in [1, 2]:
            tags_equip = []
            for champ in range(1, 6):
                col_name = f't{team}_champ{champ}id'
                hero_id = row[col_name]
                if pd.notna(hero_id):
                    hero_id = int(hero_id)
                    if hero_id in heroi_to_tags:
                        tags_equip.extend(heroi_to_tags[hero_id])
            
            tag_counter = Counter(tags_equip)
            key_parts = []
            for tag in sorted(all_tags):
                if tag in tag_counter:
                    key_parts.append(f"{tag}:{tag_counter[tag]}")
            signatura = ",".join(key_parts) if key_parts else "empty"
            
            signatura_stats[signatura]['games'] += 1
            equip_guanya = (team == 1 and winner == 1) or (team == 2 and winner == 2)
            if equip_guanya:
                signatura_stats[signatura]['wins'] += 1
    
    print(f"   Partides processades: {total_partides}")
    print(f"   Signatures amb dades: {len(signatura_stats)}")
    
    # 4. CALCULAR WIN_RATE PER TAG INDIVIDUAL
    print("\n4. CALCULANT WIN_RATE PER TAG INDIVIDUAL...")
    
    tag_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
    
    for idx, row in DataSet_herois.iterrows():
        winner = row['winner']
        for team in [1, 2]:
            tags_equip = []
            for champ in range(1, 6):
                hero_id = row[f't{team}_champ{champ}id']
                if pd.notna(hero_id):
                    hero_id = int(hero_id)
                    if hero_id in heroi_to_tags:
                        tags_equip.extend(heroi_to_tags[hero_id])
            
            equip_guanya = (team == 1 and winner == 1) or (team == 2 and winner == 2)
            for tag in set(tags_equip):
                tag_stats[tag]['games'] += 1
                if equip_guanya:
                    tag_stats[tag]['wins'] += 1
    
    tag_win_rates = {}
    for tag, stats in tag_stats.items():
        if stats['games'] > 0:
            tag_win_rates[tag] = stats['wins'] / stats['games']
        else:
            tag_win_rates[tag] = 0.5
    
    # 5. CALCULAR SCORE DE TAGS PER SIGNATURA
    print("\n5. CALCULANT SCORE DE TAGS...")
    
    signatura_scores = {}
    for signatura, stats in signatura_stats.items():
        games = stats['games']
        wins = stats['wins']
        
        if games >= min_partides:
            score = wins / games
        else:
            if signatura == "empty":
                score = 0.5
            else:
                tags_in_signatura = []
                for part in signatura.split(','):
                    tag, count = part.split(':')
                    count = int(count)
                    for _ in range(count):
                        tags_in_signatura.append(tag)
                
                tags_unics = set(tags_in_signatura)
                if tags_unics:
                    scores_tags = [tag_win_rates.get(tag, 0.5) for tag in tags_unics]
                    score = np.mean(scores_tags)
                else:
                    score = 0.5
        
        signatura_scores[signatura] = score
    
    # 6. ASSIGNAR SCORES A LES COMBINACIONS
    combo_scores = {}
    for combo_idx, signatura in combo_to_signatura.items():
        combo_scores[combo_idx] = signatura_scores.get(signatura, 0.5)
    
    # 7. AFEGIR COLUMNES A DataSet_reduit
    print("\n6. AFEGINT SCORES DE TAGS AL DATASET...")
    
    team1_scores = []
    team2_scores = []
    
    for idx, row in DataSet_reduit.iterrows():
        combo1 = row['team1_comb_index']
        combo2 = row['team2_comb_index']
        
        score1 = combo_scores.get(combo1, 0.5)
        score2 = combo_scores.get(combo2, 0.5)
        
        team1_scores.append(score1)
        team2_scores.append(score2)
    
    DataSet_reduit['team1_tag_score'] = team1_scores
    DataSet_reduit['team2_tag_score'] = team2_scores
    DataSet_reduit['tag_score_diff'] = DataSet_reduit['team1_tag_score'] - DataSet_reduit['team2_tag_score']
    
    print(f"   Columnes afegides: team1_tag_score, team2_tag_score, tag_score_diff")
    
    # 8. ESTADÍSTIQUES I RESULTATS
    print("\n7. ESTADÍSTIQUES DE SCORES DE TAGS:")
    print(f"   Team1 tag score mitjà: {DataSet_reduit['team1_tag_score'].mean():.3f}")
    print(f"   Team2 tag score mitjà: {DataSet_reduit['team2_tag_score'].mean():.3f}")
    print(f"   Diferència mitjana: {DataSet_reduit['tag_score_diff'].mean():.3f}")
    
    # Mostrar millors composicions
    signatura_list = []
    for signatura, stats in signatura_stats.items():
        games = stats['games']
        if games > 0:
            win_rate = stats['wins'] / games
            signatura_list.append({
                'signatura': signatura,
                'partides': games,
                'win_rate': win_rate,
                'score': signatura_scores.get(signatura, 0.5)
            })
    
    if signatura_list:
        signatura_df = pd.DataFrame(signatura_list)
        signatura_df = signatura_df.sort_values('win_rate', ascending=False)
        
        print("\n   TOP 5 COMPOSICIONS DE TAGS AMB MILLOR WIN RATE:")
        for idx, row in signatura_df.head(5).iterrows():
            if row['signatura'] == "empty":
                desc = "Sense tags"
            else:
                parts = []
                for part in row['signatura'].split(','):
                    tag, count = part.split(':')
                    parts.append(f"{count}x {tag}")
                desc = " + ".join(parts)
            print(f"     {desc}: WR {row['win_rate']:.1%} (score: {row['score']:.3f}) en {row['partides']} partides")
    
    return DataSet_reduit


# FLUX COMPLET PER IMPLEMENTAR TAGS AL MODEL
def implementar_tags_model(DataSet_reduit, DataSet_herois, combinacions_equips_df, min_partides=5):
    """
    Flux complet per implementar l'anàlisi de tags i afegir les columnes al model.
    """
    print("="*80)
    print("IMPLEMENTANT ANÀLISI DE TAGS AL MODEL")
    print("="*80)
    
    DataSet_amb_tags = analisi_per_tags(
        DataSet_reduit.copy(),
        DataSet_herois,
        combinacions_equips_df,
        min_partides
    )
    
    print("\n" + "="*80)
    print("IMPLEMENTACIÓ COMPLETADA")
    print("="*80)
    print(f"Dataset original ampliat amb {len(DataSet_amb_tags)} files")
    print("Columnes afegides: team1_tag_score, team2_tag_score, tag_score_diff")
    
    return DataSet_amb_tags
