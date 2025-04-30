
# (""""""""""""""""""""""Importlar""""""""""""""""""""""""""""""""")
import random
import numpy as np
import datetime
import itertools

# (""""""""""""""""""""""Importlar""""""""""""""""""""""""""""""""")

# (""""""""""""""""""""""Matris oluşturma""""""""""""""""""""""""""""""""")

an = datetime.datetime.now()
def process_node_coords(lines):
    node_coords = {}
    reading_coords = False

    for line in lines:
        if line.strip() == "NODE_COORD_SECTION" :
            reading_coords = True
            continue
        elif line.strip() == "EOF":
            break

        if reading_coords:
            parts = line.split()
            if len(parts) == 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = (x, y)

    return node_coords

def process_tsp_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
        boncuk = []
        veri_eklemeye_basla = False
        print()
        print("---------------Veri Setinin Özellikleri------------------------")
        print(lines)
        # Veri setinin tipini belirlemek için bir değişken tanımla
        edge_weight_type = None

        for line in lines:
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
                print()
                print("-------------------Edge Weight Type-----------------------")
                print("------------------" ,edge_weight_type ,"-------------------------------")
        for satir in lines:
            satir = satir.strip()
            if veri_eklemeye_basla:
                if satir == "EOF":
                    break
                boncuk.append(satir)
            if satir == "NODE_COORD_SECTION":
                veri_eklemeye_basla = True
            elif satir == "EDGE_WEIGHT_SECTION":
                veri_eklemeye_basla = True
        lines = boncuk

        num_cities = len(lines)
        print()
        print("--------Veri Setininde Bulunan Şehir Sayısı--------")
        print("----------------------*" ,num_cities ,"*----------------------")
        if edge_weight_type:
            if edge_weight_type == "EUC_2D" or "ATT":
                distance_matrix = euc_2d_matrix(lines, num_cities)

            elif edge_weight_type == "LOWER_DIAG_ROW":
                distance_matrix = lower_diag_row_matrix(lines, num_cities)

            elif edge_weight_type == "UPPER_ROW" :
                distance_matrix = upper_row_matrix(lines, num_cities)
            elif edge_weight_type == "FULL_MATRIX":
                distance_matrix = full_matrix(lines, num_cities)
            elif edge_weight_type == "UPPER_DIAG_ROW":
                distance_matrix = upper_diag_row_matrix(lines, num_cities)
            else:
                print("Kod bu formata uygun değil:", edge_weight_type)
                exit(1)
        else:
            print("EDGE_WEIGHT_TYPE bulunamadı. Kod bu formatı çalıştıramamaktadır.")
            exit(1)

    return distance_matrix

def euc_2d_matrix(lines, num_cities):
    # Verilen veri setinden EUC_2D tipindeki mesafe matrisini oluştur
    node_coords = process_node_coords(lines)
    distance_matrix = np.zeros((num_cities, num_cities))

    print()
    print("------------------Uzaklık Matrisi Üretilidi----------------------")
    # print(distance_matrix)
    for i in range(num_cities):

        for j in range(num_cities):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                data_i = lines[i].split()

                x1, y1 = float(data_i[1]), float(data_i[2])

                data_j = lines[j].split()
                x2, y2 =float(data_j[1]), float(data_j[2])
                distance_matrix[i][j] = float(f"{np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2):.0f}")

    # float(f"{np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2):.1f}")
    print()
    print("------------------------Uzaklık Matrisi Dolduruldu-----------------------")
    print(distance_matrix)
    return distance_matrix


def lower_diag_row_matrix(lines, num_cities):
    # Düşük üçgen veri setinden mesafe matrisini oluştur
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        data = list(map(int, lines[i].split()))
        for j in range(i):
            distance_matrix[i][j] = data[j]

    for i in range(num_cities):
        for j in range(i, num_cities):
            distance_matrix[i][j] = distance_matrix[j][i]

    return distance_matrix


def upper_row_matrix(lines, num_cities):
    # Üst üçgen veri setinden mesafe matrisini oluştur
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        data = list(map(int, lines[i].split()))
        for j in range(i, num_cities):
            distance_matrix[i][j] = data[j - i]

    return distance_matrix


def full_matrix(lines, num_cities):
    # Tam matris veri setinden mesafe matrisini oluştur
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        data = list(map(int, lines[i].split()))
        distance_matrix[i] = data

    return distance_matrix


def upper_diag_row_matrix(lines, num_cities):
    # Üst üçgen veri setinden mesafe matrisini oluştur
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        data = list(map(int, lines[i].split()))
        for j in range(i, num_cities):
            distance_matrix[i][j] = data[j - i]

    return distance_matrix


# (""""""""""""""""""""""Matris oluşturma Bitti""""""""""""""""""""""""""""""""")
# (""""""""""""""""""""""ALTERNATİF BAŞLANGIÇ ÇÖZÜM """""""""""""""""""""""""""""""""")
def nearest_neighbor_tsp_custom(distance_matrix):
    # Verilen bir mesafe matrisi ile en yakın komşu TSP çözümü oluşturması
    num_cities = len(distance_matrix)
    unvisited_cities = list(range(num_cities))  # Ziyaret edilmemiş şehirleri bir liste olarak tutulur
    current_city = 0
    # Rassal bir başlangıç noktası seçilir
    current_city = random.choice(unvisited_cities)
    tour = [current_city + 1]  # Turumuzu oluşturan şehirleri tutacak bir liste başlangıç şehiri ile başlatır
    unvisited_cities.remove(current_city)  # Başlangıç şehrini ziyaret edilmiş şehirler listesinden kaldırma işlemi

    while unvisited_cities:
        # Henüz ziyaret edilmemiş şehirler varken döngüyü sürdür yoksa döngüde çık
        # Şu anki şehirden en yakın olanı bul
        nearest_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])

        # En yakın şehiri tur listesine ekle ve ziyaret edilmişler listesinden çıkar
        tour.append(
            nearest_city + 1)  # +1, şehirlerin indeksini 0'dan başlayan dilimlemeden 1'den başlayan indeksleme biçimine dönüştürür
        unvisited_cities.remove(nearest_city)

        current_city = nearest_city  # Şu anki şehri güncelle

    # Turun uzunluğunu hesapla calculate_tour_length fonksiyonuna giderek bu işlem gerçekleşir
    tour_length = calculate_tour_length(tour, distance_matrix)

    return tour, tour_length


def generate_greedy_solution_inside(current_tour, distance_matrix):
    num_cities = len(distance_matrix)
    unvisited_cities = set(range(num_cities))
    current_city = current_tour[0]
    tour = [current_city + 1]
    unvisited_cities.remove(current_city)

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])
        tour.append(nearest_city + 1)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    return tour


def calculate_total_distance(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i] - 1][tour[i + 1] - 1]
    return total_distance


def generate_greedy_solution(distance_matrix):
    num_cities = len(distance_matrix)  # Şehir sayısını al
    unvisited_cities = set(range(num_cities))  # Ziyaret edilmemiş şehirlerin kümesini oluştur
    current_city = random.choice(list(unvisited_cities))  # Başlangıç şehirini rastgele seç
    tour = [current_city + 1]  # Başlangıç turunu oluştur ve şehir numaralarını 1 artır
    unvisited_cities.remove(current_city)  # Başlangıç şehirini ziyaret edilmiş olarak işaretle

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])
        # Ziyaret edilmemiş şehirler arasında en yakın olanı bul
        tour.append(nearest_city + 1)  # Tura en yakın şehiri ekle ve şehir numarasını 1 artır
        unvisited_cities.remove(nearest_city)  # En yakın şehiri ziyaret edilmiş olarak işaretle
        current_city = nearest_city  # Gezginin konumunu güncelle

    return tour


def generate_swap_solution(distance_matrix):
    num_cities = len(distance_matrix)
    tour = list(range(1, num_cities + 1))

    # Rastgele iki şehir seç ve swap işlemi uygula
    i, j = random.sample(range(num_cities), 2)  # Rassal iki indis seç
    tour[i], tour[j] = tour[j], tour[i]  # Swap işlemi

    return tour


def generate_insertion_solution(distance_matrix):
    num_cities = len(distance_matrix)
    tour = list(range(1, num_cities + 1))

    # Rastgele iki şehir seç ve insertion işlemi uygula
    i, j = random.sample(range(num_cities), 2)  # Rassal iki indis seç
    selected_city = tour.pop(i)  # Seçilen şehiri çıkar
    tour.insert(j, selected_city)  # İstenilen konuma ekle

    return tour


def generate_2opt_solution(distance_matrix):
    num_cities = len(distance_matrix)
    tour = list(range(1, num_cities + 1))

    # Rastgele iki şehir seç ve 2-Opt işlemi uygula
    i, j = sorted(random.sample(range(num_cities), 2))  # Rassal iki indis seç ve sırala
    tour[i:j + 1] = reversed(tour[i:j + 1])  # 2-Opt işlemi

    return tour


# (""""""""""""""""""""""ALTERNATİF BAŞLANGIÇ ÇÖZÜM BİTTİ " """""""""""""""""""""""""""""""""")
# (""""""""""""""""""""""Intensification ve Diversification Fonksiyonları""""""""""""""""""""")
#################Kullanılmıyor Ama Güzel Olabilir#################
def diversification_strategy(current_tour):
    num_cities = len(current_tour)

    # Kaç şehiri yer değiştireceğimizi belirle
    num_swaps = random.randint(1,
                               min(3, num_cities - 1))  # Max 3 şehir yer değiştirsin, ancak turda en az 2 şehir olmalı

    # Belirlenen sayıda şehiri yer değiştirerek turu güncelle
    for _ in range(num_swaps):
        # Rastgele iki farklı indis seç
        i, j = random.sample(range(num_cities), 2)

        # İki şehri yer değiştirerek turu güncelle
        current_tour[i], current_tour[j] = current_tour[j], current_tour[i]

    return current_tour


#################Kullanılmıyor Ama Güzel Olabilir#################

def hill_climbing(tour, distance_matrix):
    # Hill Climbing algoritması, mevcut turu iteratif olarak iyileştirmeye çalışır.
    improved = True
    while improved:
        improved = False

        # Tüm şehir çiftleri üzerindeki değişimleri kontrol etmek için iki iç içe geçmiş döngü.
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                # İki şehir arasındaki mesafeleri karşılaştırarak değişiklik yapılıp yapılamayacağını kontrol et.

                # Eğer mevcut değişiklik, mevcut turu iyileştiriyorsa:
                if distance_matrix[tour[i - 1] - 1][tour[j - 1] - 1] + distance_matrix[tour[i] - 1][tour[j] - 1] < \
                        distance_matrix[tour[i - 1] - 1][tour[i] - 1] + distance_matrix[tour[j - 1] - 1][tour[j] - 1]:
                    # İlgili şehirleri yer değiştirerek turu güncelle.
                    tour[i:j] = tour[i:j][::-1]

                    improved = True  # Değişiklik yapıldı, bu yüzden iyileştirme devam ediyor.

    return tour


def variable_neighborhood_search(tour, tour_lenght, distance_matrix, max_neighborhood):
    best_tour = tour.copy()
    best_tour_length = tour_lenght
    for neighborhood_size in range(1, max_neighborhood + 1):
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - neighborhood_size + 1):
                new_tour = best_tour.copy()
                # Belirli bir komşuluk boyutuna göre tur segmentini ters çevirme
                j = i + neighborhood_size
                if j < len(tour):
                    new_tour[i:j] = new_tour[i:j][::-1]

                    # new_tour_length = calculate_tour_length(new_tour, distance_matrix)
                    #
                    ## Yeni turun uzunluğunu kontrol et ve daha kısa ise güncelle
                    # if new_tour_length < best_tour_length:
                    #    best_tour = new_tour
                    #    best_tour_length = new_tour_length
                    #    improved = True
                    #    break  # En iyi çözümü bulduktan sonra döngüyü kır
                    change_in_length = calculate_change_in_length(best_tour, i, j, distance_matrix)
                    #
                    # Yeni turun uzunluğunu kontrol et ve daha kısa ise güncelle
                    if tour_lenght + change_in_length < best_tour_length:
                        best_tour = new_tour
                        best_tour_length += change_in_length
                        improved = True
                        break  # En iyi çözümü bulduktan sonra döngüyü kır

    return best_tour, best_tour_length


def calculate_change_in_length(tour, i, j, distance_matrix):
    """
    Verilen bir turda, belirli bir i ve j arasındaki değişikliğin tur uzunluğuna olan etkisini hesaplar.
    """
    prev_i = tour[i - 1]
    next_i = tour[i]
    prev_j = tour[j - 1]
    next_j = tour[j]

    # Mevcut i ve j konumlarındaki mesafeler
    current_distance = distance_matrix[prev_i - 1][next_i - 1] + distance_matrix[prev_j - 1][next_j - 1]

    # Yer değiştirdiğimizde oluşacak yeni mesafeler
    swapped_distance = distance_matrix[prev_i - 1][prev_j - 1] + distance_matrix[next_i - 1][next_j - 1]

    # Değişikliğin tur uzunluğuna olan etkisini hesapla
    change_in_length = swapped_distance - current_distance

    return change_in_length


def adjust_tabu_tenure(stagnation_counter, base_tabu_tenure, max_tabu_tenure):
    """
    Stagnation süresine bağlı olarak tabu tenure'yi ayarlar.

    :param stagnation_counter: Algoritmanın iyileşme göstermediği iterasyon sayısı.
    :param base_tabu_tenure: Tabu tenure'nin başlangıç değeri.
    :param max_tabu_tenure: Tabu tenure'nin maksimum değeri.
    :return: Ayarlanmış tabu tenure.
    """

    if stagnation_counter < 100:
        return base_tabu_tenure
    elif stagnation_counter < 200:
        return min(base_tabu_tenure + 50, max_tabu_tenure)
    elif stagnation_counter < 300:
        return min(base_tabu_tenure + 100, max_tabu_tenure)
    else:
        return max_tabu_tenure


# (""""""""""""""""""""""Intensification ve Diversification Fonksiyonları Bitti""""""""""""")


# (""""""""""""""""""""""TABU ÇÖZÜMDE KULLANILAN ELEMANLAR BAŞLANGICI """"""""""""""""""""""""""""""""")


def calculate_tour_length(tour, distance_matrix):
    # Tur uzunluğu hesaplama fonksiyonu

    # Tur uzunluğunu saklamak için bir değişken oluşturuyoruz.
    tour_length = 0

    # Her iki şehir arasındaki mesafeyi hesaplamak için turdaki şehirler arasında bir döngü oluşturuyoruz.
    for i in range(len(tour) - 1):
        # İki ardışık şehir arasındaki mesafeyi toplam uzunluğa ekliyoruz.
        # Not: `tour` dizisindeki şehirler 1'den başladığı için, diziyi kullanırken bir çıkartma yapıyoruz.
        # Örneğin, tur[0] aslında şehir 1'i temsil eder.
        tour_length += distance_matrix[tour[i] - 1][tour[i + 1] - 1]

    # Son şehir ile başlangıç şehiri arasındaki mesafeyi de hesaba katmalıyız.
    # Bu, turun tam bir döngü olduğu anlamına gelir, bu yüzden son şehirden başlangıç şehrine gitmeliyiz.
    tour_length += distance_matrix[tour[-1] - 1][tour[0] - 1]

    # Sonuç olarak, turun toplam uzunluğunu hesapladığımız bu değişkeni döndürüyoruz.
    return tour_length


def generate_initial_solution(num_cities):
    return random.sample(range(1, num_cities + 1), num_cities)  # Rassal bir başlangıç turu oluşturur


def is_tabu(tabu_list, move):
    return move in tabu_list  # Eğer hareket tabu listesindeyse True döndürür


def calculate_move_impacts(current_tour, move, distance_matrix):
    i, j = move
    # İki şehir arasındaki mesafeleri al

    before_move_distance = distance_matrix[current_tour[i - 1] - 1][current_tour[i] - 1] + \
                           distance_matrix[current_tour[j - 1] - 1][current_tour[j] - 1]
    after_move_distance = distance_matrix[current_tour[i - 1] - 1][current_tour[j] - 1] + \
                          distance_matrix[current_tour[i] - 1][current_tour[j - 1] - 1]
    # Hareketin toplam uzunluktaki değişikliği
    impact = after_move_distance - before_move_distance  # Hareketin etkisi hesaplanır
    return impact


def generate_all_moves_chunks(num_cities, num_chunks):
    all_moves = list(itertools.combinations(range(num_cities), 2))  # Tüm hareketler oluşturulur
    chunk_size = len(all_moves) // num_chunks  # Her iterasyonda kullanılacak hareket sayısı hesaplanır
    return [all_moves[i:i + chunk_size] for i in range(0, len(all_moves), chunk_size)]  # Hareketler parçalara ayrılır


# (""""""""""""""""""""""TABU ÇÖZÜMDE KULLANILAN ELEMANLAR BİTİŞİ """"""""""""""""""""""""""""""""")
# (""""""""""""""""""""""TABU SEARCH ÇÖZÜM BAŞLANGICI """"""""""""""""""""""""""""""""")

def tabu_search_tsp_optimized(distance_matrix, max_iterations, base_tabu_tenure, num_chunks, max_neighborhood, stop_value):
    num_cities = len(distance_matrix)
    function_evaluations = 0
    best_lengths_over_iterations = []
    initial_solution_type = "random"  # Başlangıç çözümü türü seçimi (greedy, swap, insertion, 2opt, near, random)
    print((max_neighborhood))
    # Başlangıç çözümüne göre ilk tur oluşturulur
    if initial_solution_type == "greedy":
        current_tour = generate_greedy_solution(distance_matrix)
    elif initial_solution_type == "swap":
        current_tour = generate_swap_solution(distance_matrix)
    elif initial_solution_type == "insertion":
        current_tour = generate_insertion_solution(distance_matrix)
    elif initial_solution_type == "2opt":
        current_tour = generate_2opt_solution(distance_matrix)
    elif initial_solution_type == "near":
        current_tour = nearest_neighbor_tsp_custom(distance_matrix)[0]
    elif initial_solution_type == "random":
        current_tour = generate_initial_solution(num_cities)
    else:
        raise ValueError("Geçersiz başlangıç çözümü türü")

    # Başlangıç turunun uzunluğu hesaplanır
    current_tour_length = calculate_tour_length(current_tour, distance_matrix)
    print("Başlangıç turu uzunluğu:", current_tour_length)
    # En iyi tur ve uzunluğu başlangıç turu olarak ayarlanır
    best_tour = current_tour.copy()
    best_tour_length = current_tour_length
    best_lengths_over_iterations.append(best_tour_length)
    # Tabu listesi ve diğer yardımcı değişkenler başlatılır
    tabu_list = []
    stagnation_counter = 0
    base_tabu_tenure = max_neighborhood * 10
    # Stagnation counter intensification ve diversification kullanımını
    # belli bir durağınlıktan sonra uygulamak için burada bulunuyor
    max_tabu_tenure = (max_neighborhood * max_neighborhood / 2 )-10
    all_moves_chunks = generate_all_moves_chunks(num_cities, num_chunks)

    # Belirlenen maksimum iterasyon sayısı kadar Tabu Search döngüsü
    for iteration in range(max_iterations):
        # Mevcut turun tüm komşuları oluşturulur
        # an = datetime.datetime.now()
        neighbors = []
        neighbors_best = []

        tabu_tenure = adjust_tabu_tenure(stagnation_counter, base_tabu_tenure, max_tabu_tenure)
        current_all_moves = all_moves_chunks[
            iteration % num_chunks]  # Her iterasyonda farklı bir komşu kümesi oluşturulur

        # Her komşu için hareketin etkisi hesaplanır ve listeye eklenir

        # En Uzun Süren bölüm bu hareket üretme ve komşuların hareketlerini impactleme move'ları azaltmaya gidebiliriz.
        for move in current_all_moves:  # Her hareket için
            i, j = move
            neighbor = current_tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap işlemi
            move_tuple = (i, j)  # Hareket tuple olarak tutulur

            if not is_tabu(tabu_list, move_tuple):  # Eğer hamle tabu listesinde değilse
                move_impact = calculate_move_impacts(current_tour, move, distance_matrix)  # Hareketin etkisi hesaplanır
                if move_impact < 0:  # Eğer hareketin etkisi negatifse
                    neighbors.append((neighbor, move_tuple, move_impact))  # Komşulara eklenir
                elif move_impact < best_tour_length * 0.1:  # Eğer hareketin etkisi en iyi turun %10'undan küçükse
                    neighbors.append((neighbor, move_tuple, move_impact))  # Komşulara eklenir

        # print("hareketler oluşturuldu")
        # Eğer komşu yoksa döngüden çıkılır
        if not neighbors:
            break

        # En küçük etkiye sahip olan komşu seçilir
        best_neighbor, best_move, best_move_impact = min(neighbors, key=lambda x: x[2])  # En iyi komşu seçilir

        # Tabu kriteri kontrol edilir ve en iyi komşu kabul edilir veya reddedilir

        current_tour = best_neighbor
        current_tour_length = calculate_tour_length(current_tour, distance_matrix)
        # Eğer güncel tur daha iyi ise en iyi tur güncellenir, aksi takdirde bir artış sayacı arttırılır
        if current_tour_length < best_tour_length:  # Eğer güncel tur daha iyi ise
            print("En iyi tur ile aynı mesafeye ait ya da daha iyi bir tur elde edildi")
            best_tour = current_tour.copy()  # En iyi tur güncellenir
            best_tour_length = current_tour_length  # En iyi tur uzunluğu güncellenir
            # Stagnation counter sıfırlanır

        else:
            stagnation_counter += 1  # Değilse Stagnation counter arttırılır
        # Tabu listesine eklenen hamle

        tabu_list.append(best_move)  # En iyi hamle tabu listesine eklenir

        # Tabu listesinin uzunluğu kontrol edilir ve gerekiyorsa eski hamleler silinir
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        # Intensificatio
        hill_tour = hill_climbing(current_tour, distance_matrix)
        a = calculate_tour_length(hill_tour, distance_matrix)
        if a < best_tour_length:  # Eğer hill en iyi turu bulduysa
            print("iyileştirme sağlandı ")
            best_tour = hill_tour.copy()  # En iyi tur güncellenir
            current_tour = hill_tour.copy()  # Mevcut tur güncellenir
            best_tour_length = calculate_tour_length(current_tour, distance_matrix)  # En iyi tur uzunluğu güncellenir
            stagnation_counter = 0  # Stagnation counter sıfırlanır
            max_neighborhood = int(1 * len(distance_matrix))
            print(best_tour_length)


        vns_tour, vns_tour_length = variable_neighborhood_search(current_tour, current_tour_length, distance_matrix,
                                                                 max_neighborhood)
        if vns_tour_length < best_tour_length:
            print("iyileştirme sağlandı ")
            best_tour = vns_tour.copy()
            current_tour = vns_tour.copy()
            best_tour_length = vns_tour_length
            stagnation_counter = 0
            max_neighborhood = int(1 * len(distance_matrix))

        best_lengths_over_iterations.append(best_tour_length)
        # Diversification
        if stagnation_counter > 500:
            if num_chunks > 1:  # Eğer chunk sayısı 1'den büyükse
                num_chunks = int(num_chunks / 2)  # Hareketlerin parçalara ayrılma sayısı 1'e düşürülür
                print("Chunklar birleştirildi- chunk sayısı:", num_chunks)
                all_moves_chunks = generate_all_moves_chunks(num_cities, num_chunks)
                stagnation_counter = 0  # Stagnation counter sıfırlanır
        if stagnation_counter > 200:
            stagnation_counter = 0  # Stagnation counter sıfırlanır
            if max_neighborhood * 2 < len(distance_matrix):
                max_neighborhood = int(max_neighborhood * 2)
            elif max_neighborhood * 2 > len(distance_matrix):
                max_neighborhood = len(distance_matrix)
        if stop_value == best_tour_length:
            print(iteration, "İterasyon Sayısı", "En iyi tur uzunluğu", best_tour_length, "Çalışma süresi:", an1 - an,
                  "move impact eleman sayısı", len(neighbors), "en iyi komşu", best_move_impact, "staga",
                  stagnation_counter, "max ne", max_neighborhood, "tabu liste boyutu",
                  len(tabu_list))  # En iyi tur uzunluğu yazdırılır
            break
        an1 = datetime.datetime.now()
        if iteration % 100 == 0:
            # Her iterasyon sonunda durumu yazdır
            print(iteration, "İterasyon Sayısı", "En iyi tur uzunluğu", best_tour_length, "Çalışma süresi:", an1 - an,
                  "move impact eleman sayısı", len(neighbors), "en iyi komşu", best_move_impact, "staga",
                  stagnation_counter, "max ne", max_neighborhood, "tabu liste boyutu",
                  len(tabu_list), 'gap',(best_tour_length/stop_value)-1)  # En iyi tur uzunluğu yazdırılır
            function_evaluations += 1
    # an1 = datetime.datetime.now()
    # print("Çalışma süresi:", an1 - an)
    # Toplam değerlendirme sayısı yazdırılır
    print(f"Toplam Tabu Search algoritması için fonksiyon değerlendirmesi sayısı: {function_evaluations}")

    # En iyi tur ve uzunluğu döndürülür
    return best_tour, best_tour_length, best_lengths_over_iterations






# (""""""""""""""""""""""TABU SEARCH ÇÖZÜM BİTİŞİ """"""""""""""""""""""""""""""""")
def ask_user_to_reorder_tour(best_tour):
    print("Optimal tur: ", best_tour)
    answer = "yes"
    if answer == "yes":
        # 1 numaralı şehir başa alınıyor
        index_of_city_1 = best_tour.index(1)
        best_tour = best_tour[index_of_city_1:] + best_tour[:index_of_city_1]
    return best_tour


# (""""""""""""""""""""""KULLANICI VERİLERİ """"""""""""""""""""""""""""""""")
# Kullanım örneği
max_iterations = 20000  # Maksimum iterasyon sayısı
base_tabu_tenure = 760  # Tabu tenure uzunluğu
num_chunks = 1  # Hareketlerin parçalara ayrılma sayısı
stop_value = 7542  # Döngüyü durdurma değeri
file_name = "berlin52.tsp"

# ("""""""""""""""""""""ÇALIŞTIRMA SATIRLARI """"""""""""""""""""""""""""""""")
distance_matrix = process_tsp_file(file_name)
max_neighborhood = int(1 * len(distance_matrix))
best_tour_tabu_all, best_tour_length_tabu_all, best_tour_over_iterations_all = tabu_search_tsp_optimized(distance_matrix, max_iterations, base_tabu_tenure, num_chunks, max_neighborhood,stop_value)
an1 = datetime.datetime.now()
print("Çalışma süresi:", an1 - an)
best_tour = ask_user_to_reorder_tour(best_tour_tabu_all)

# Son haliyle optimal turu yazdır

#(""""""""""""""""""""" """"""""""""""""""""""""""""""""")
print("Tabu Search - En iyi tur uzunluğu:", best_tour_length_tabu_all)
print("Son haliyle optimal tur: ", best_tour)

###### METASEZGİSEL FİNAL RAPORUNDA ELDE EDİLEN SONUÇLARIN GRAFİKLERİNİN ÇİZİMİ İÇİN KULLANILMIŞTIR ######
'''''
plt.plot( best_tour_over_iterations_all, label='Hill+Tabu')

plt.legend()
plt.xlabel('Iterasyon Sayısı')
plt.ylabel('En İyi Tur Uzunluğu')
plt.title('İterasyonlardaki En İyi Tur Uzunluğu')
plt.show()

# ("""""""""""""""""""""ÇALIŞMA SÜRESİ """"""""""""""""""""""""""""""""")


# ("""""""""""""""""""""ELDE EDİLEN ÇÖZÜM GRAFİĞİ """"""""""""""""""""""""""""""""")

with open(file_name) as f:
    problem = tsplib95.read(f)

nodes = list(problem.node_coords.values())
node_x = []
node_y = []

for i in nodes:
    node_x.append(i[0])
    node_y.append(i[1])
plt.scatter(node_x, node_y, marker='s', s=5)
plt.title('Nodeların X ve Y Koordinatındaki Konumları', fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=16, fontweight='bold')
plt.ylabel('Y', fontsize=16, fontweight='bold')
plt.show()
route_node_x = []
route_node_y = []

# En iyi çözümdeki düğümlerin x ve y koordinatları çıkartılıyor

for i in best_tour_tabu_all:
    route_node_x.append(node_x[i - 1])
    route_node_y.append(node_y[i - 1])

route_node_x.append(node_x[best_tour_tabu_all[0] - 1])
route_node_y.append(node_y[best_tour_tabu_all[0] - 1])
fig, (best) = plt.subplots(1, 1, figsize=(5, 5))
best.plot(route_node_x, route_node_y, c='r')
best.set_title('Bulunan En İyi Çözüm Rotası = ' + str(best_tour_length_tabu_all))
plt.show()
#################   OPTIMAL ÇÖZÜMLER  #################
Name #cities Type Bounds
a280  2579
ali535 202310
att48 10628
att532 27686
bayg29 1610
bays29 2020
berlin52 7542
bier127 118282
brazil58 25395
brd14051 [468942,469445]
brg180  1950
burma14  3323
ch130 6110
ch150 6528
d198  15780
d493  35002
d657  48912
d1291  50801
d1655  62128
d2103  [79952,80450]
d15112 [1564590,1573152]
d18512  [644650,645488]
dantzig42  699
dsj1000 18659688
eil51  426
eil76  538
eil101  629
fl417  11861
fl1400  20127
fl1577  [22204,22249]
fl3795  [28723,28772]
fnl4461  182566
fri26  937
gil262  2378
gr17  2085
gr21  2707
gr24  1272
gr48  5046
gr96  55209
gr120  6942
gr137  69853
gr202  40160
gr229  134602
gr431  171414
gr666 294358
hk48 11461
kroA100  21282
kroB100  22141
kroC100  20749
kroD100  21294
kroE100  22068
kroA150  26524
kroB150  26130
kroA200  29368
kroB200  29437
lin105 14379
lin318 42029
linhp318  41345
nrw1379  56638
p654  34643
pa561  2763
pcb442  50778
pcb1173  56892
pcb3038  137694
pla7397  23260728
pla33810  [65913275,66116530]
pla85900  [141904862,142487006]
pr76  108159
pr107  44303
pr124  59030
pr136  96772
pr144  58537
pr152  73682
pr226  80369
pr264  49135
pr299  48191
pr439  107217
pr1002  259045
pr2392  378032
rat99  1211
rat195  2323
rat575  6773
rat783  8806
rd100  7910
rd400  15281
rl1304  252948
rl1323  270199
rl1889  316536
rl5915  [565040,565530]
rl5934  [554070,556045]
rl11849  [920847,923368]
si175 21407
si535  48450
si1032  92650
st70  675
swiss42  1273
ts225  126643
tsp225  3919
u159  42080
u574 36905
u724  41910
u1060  224094
u1432  152970
u1817  57201
u2152  64253
u2319  234256
ulysses16  6859
ulysses22  7013
usa13509  [19947008,19982889]
vm1084  239297
vm1748  336556
'''