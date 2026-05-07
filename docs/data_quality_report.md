# Veri Kalite Raporu — Edge-IIoTset

## 1. Veri Seti Genel Bilgisi

| Özellik | Değer |
|:---|:---|
| **Veri Seti** | Edge-IIoTset — IoT/IIoT Cyber Security Dataset |
| **Kaynak** | Kaggle (Mohamed Amine Ferrag) |
| **Kullanılan Dosya** | `ML-EdgeIIoT-dataset.csv` (~82 MB) |
| **Kolon Sayısı** | 63 |
| **Saldırı Tipleri** | 14+ (DDoS, MITM, XSS, SQL Injection, Backdoor vb.) |
| **Hedef Değişken** | `Attack_label` (0=Normal, 1=Attack) ve `Attack_type` |

## 2. Kolon Kategorileri

| Kategori | Kolonlar | Açıklama |
|:---|:---|:---|
| Zaman | `frame.time` | Paket zaman damgası |
| IP Adresleri | `ip.src_host`, `ip.dst_host` | Kaynak ve hedef IP |
| ARP | `arp.*` (4 kolon) | ARP protokol bilgileri |
| ICMP | `icmp.*` (4 kolon) | ICMP mesaj bilgileri |
| HTTP | `http.*` (9 kolon) | HTTP istek/yanıt bilgileri |
| TCP | `tcp.*` (16 kolon) | TCP bağlantı bilgileri |
| UDP | `udp.*` (3 kolon) | UDP akış bilgileri |
| DNS | `dns.*` (7 kolon) | DNS sorgu bilgileri |
| MQTT | `mqtt.*` (13 kolon) | IoT MQTT protokol bilgileri |
| Modbus | `mbtcp.*` (3 kolon) | Endüstriyel protokol |
| Etiketler | `Attack_label`, `Attack_type` | Sınıflandırma hedefleri |

## 3. Tespit Edilen Veri Kalite Sorunları

### 3.1 Eksik Değerler
Veri setindeki çoğu kolon IoT protokollerine aittir. Bir paket HTTP ise MQTT kolonları doğal olarak 0/boş olur. Bu beklenen bir durumdur ve eksik değer olarak değil, "aktivite yok" olarak ele alınmalıdır.

**Strateji:** Sayısal eksik değerler → 0 ile dolduruldu (ağ trafiğinde 0 = aktivite yok)

### 3.2 Sonsuz (Inf) Değerler
Bazı sayısal kolonlarda sıfıra bölme sonucu `inf` değerler oluşabilir.

**Strateji:** Inf değerler → null'a çevrildi → 0 ile dolduruldu

### 3.3 Gereksiz Kolonlar
ARP ve ICMP protokol kolonları çoğunlukla 0 değerine sahiptir ve ML modeline katkısı minimumdur.

**Strateji:** `arp.*` ve `icmp.*` kolonları (8 adet) kaldırıldı

## 4. Feature Engineering — 5 Yeni Özellik

| # | Feature | Formül | Tespit Ettiği Saldırı | Gerekçe |
|:--|:---|:---|:---|:---|
| 1 | `traffic_asymmetry_ratio` | tcp.ack / (tcp.seq + 1) | DDoS Flood | SYN flood'da ACK/SEQ oranı bozulur |
| 2 | `pkt_size_cv` | abs(tcp.len - tcp.payload) / (tcp.len + 1) | Port Scan, Vuln Scanner | Saldırı trafiğinde paket boyutu değişken |
| 3 | `flow_intensity` | tcp.len × tcp.flags | DDoS UDP/ICMP | Volumetrik saldırılar büyük+bayraklı paketler gönderir |
| 4 | `iat_regularity` | udp.time_delta / (frame.time + 1) | Botnet, Brute Force | Bot'lar düzenli aralıkla gönderir |
| 5 | `conn_efficiency` | tcp.syn / (tcp.fin + tcp.rst + 1) | Port Scan, OS Fingerprint | Keşif saldırıları bağlantıyı tamamlamaz |

## 5. Delta Lake Katmanları

| Katman | Path | İçerik |
|:---|:---|:---|
| **Silver** | `delta-storage/silver/network_traffic/` | Temizlenmiş veri (gereksiz kolonlar çıkarılmış, eksik/inf doldurulmuş, duplikatlar kaldırılmış) |
| **Gold** | `delta-storage/gold/ml_ready/` | Silver + 5 yeni feature eklenmiş, ML'ye hazır |

## 6. Sonuç

Veri seti IoT ağ trafiği paketlerinden oluşmaktadır. Her satır bir ağ paketi veya akışını temsil eder.
63 protokol kolonunun çoğu belirli protokol türlerine özgüdür (sadece HTTP paketlerinde http.* dolu olur).
Bu durum "eksik değer" değil, veri setinin doğal yapısıdır.

Temizleme ve feature engineering sonrasında veri, Adım 6'daki 5 ML modelini eğitmek için hazır durumdadır.
