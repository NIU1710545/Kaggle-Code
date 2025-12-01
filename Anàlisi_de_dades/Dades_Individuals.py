

def analitzar_percentatges_victoria_per_durada(duplicate_durations):
    # Comptabilitzar per cada nombre de partides duplicades, el percentatge de guanys de cada equip 
    duration_counts = duplicate_durations['gameDuration'].value_counts()
    for duration, count in duration_counts.items():
        subset = duplicate_durations[duplicate_durations['gameDuration'] == duration]
        winner_counts = subset['winner'].value_counts()
        print(f"\nDurada de la partida: {duration} segons - Nombre de partides: {count}")
        for team, wins in winner_counts.items():
            win_percentage = (wins / count) * 100
            print(f"Equip {team} - Partides guanyades: {wins} ({win_percentage:.2f}%)")

