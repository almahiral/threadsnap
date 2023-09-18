import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
from get_twt_thread import get_thread, get_text
import html.parser
from html.parser import HTMLParser
from functools import partial
import pandas as pd
import numpy as np
from indoNLP.preprocessing import replace_slang, replace_word_elongation
from indoNLP.preprocessing.slang_data import SLANG_DATA
import emoji
import pickle
from semantic_search import do_semantic_search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ========== 1. Fetch the array of tweets (done) ==========
'''Fetch Tweets from Twitter URL (input from user) using Twitter API.'''
thread0 = [
  "Nonton 'Call Me Chihiro' rasanya  kek mendapat   dekapan erat &amp; hangat dari orang terkasih yg bikin hati nyaman sekaligus senyum mengembang lebar. \n\nIni kisah seorang mantan PSK yg kerja di warung bento yg kebaikan hatinya mengubah hidup orang-orang terbuang, kesepian &amp; merana :( https://t.co/ROK9oJogC5",
  "@kasumiii Ku menikmatiiiiiiii bgt film ini 😍😎😎 (meski mengalun pelan dengan durasi lebih dari 2 jam) karena mendamaikan menyaksikan interaksi manusia-manusia baik yg saling menguatkan dengan bercanda ria atau ngumpul bareng seraya bertukar pikiran tentang makna kehidupan :) https://t.co/OVeYAGTXqD",
  "Ya, selalu ada tempat di hatiku buat film tentang orang baik... Si tokoh utama yg terbiasa disalahpahami mengerti rasanya terbuang",
  "Saat jalani hidup baru, dia memilih berbagi kebahagiaan pada sesama dengan bersikap ramah &amp; jadi sosok manusia yg siap mendengar serta ulurkan tangan https://t.co/nquW1OSVyq",
  "Lanjut nanti ya guys, nunggu rame dulu hehehe...",
  "#Kasumi #Arimura bermain simpatik sebagai seseorang yg sepintas tampak jalani hidup penuh suka cita, tapi diam-diam menyimpan rahasia kelam yg membuat hatinya kerap dirundung kehampaan &amp; sepi . Colek @kasumiarimura hehehe :3 . Namun melalui penderitaan serta pertemuan dengan orang-orang baik, dia belajar berempati. Seru gak sih???? https://t.co/bb70Ee6Fab",
  "Ini memang bukan tontonan healing yg bisa ditonton nyaman bareng seluruh keluarga!!! Karena ada muatan seks & obrolan soal bunuh diri. Pun begitu, ini tipe film yg akan relate bagi yg sedang pertanyakan tujuan hidup &amp; dibutuhkan dunia sekarang ini karena ajakannya menebar kebaikan. Aku kasih skor 10 &gt; 3!!!! bgs bgt! https://t.co/6c2wy6Unus",
  "gais rame banget nih gaada yang mau mutualan?❤️ https://t.co/N6njDMaJBz"
]

thread1 = [
    "cerita soal privilege dan pilihan hidup. - sebuah utas panjang.. lulus stan. ipk tinggi. masuk kantor pusat adalah impian banyak orang. looks like livin in a dream, right? kayanya nyaman dan karir lancar jaya. namun, ada orang yang memilih hal lain. meninggalkan privilege jakarta dan kantor pusat dengan beragam impian. pindah ke luar jawa. tak lama menyusul suaminya, si suami dimutasi, satu kanwil tetapi beda provinsi. ini namanya pilihan hidup. tapi ada satu privilige : privilige hidup bersatu bersama keluarga. bisa jadi kantor pusat adalah privilge, ipk terbaik pun privilege. tapi privilege dan prioritas orang beda-beda. dan barangkali bersatu bersama keluarga tercinta juga privilige dan prioritas yang diutamakan. 👌. satu teman lagi ipk tinggi. homebase. tentu itu privilege, tidak semua orang bisa langsung ke homebase. baginya tujuan sudah tercapai, homebase. ia tidak mengejar apa-apa lagi. tidak mengejar karir, tidak mengejar jabatan, tidak ngoyo di pendidikan. homebase saja. maka ia memilih memanfaatkan privilege yang tak semua orang bisa. dan itu pilihan ia tidak melanjutkan kuliah, ditawari promosi dengan halus menolak dan menyilakan orang lain yang lebih ingin dan butuh. ia bilang ""kalau semua orang mau jadi bos, lalu siapa yang bisa laden?"" karir, pendidikan, pangkat, jabatan adalah leverage yang bisa mendatangkan banyak privilege bagi seseorang. banyak orang yg berjuang berdarah-darah untuk mendapatkannya. namun, ada juga yang memilih : ketentraman, urip tenang. pangkat boleh rendah tetapi hati ayem setiap waktu. satu kawan saya resign, salah satu yang resign paling awal. mengagetkan sekali. peringkat tinggi, penempatan di KPP bergengsi. dengan ikhlas, tulus, mengundurkan diri dari PNS demi membersamai suaminya di ujung barat Indonesia. suaminya penempatan di sana. si istri tentu ikut juga. resign adalah pilihan untuk bisa membersamai suami. bersatu bersama keluarga adalah pilihan yang diambil. memang salah satu harus mengalah, tetapi ada privilege lain yang barangkali tidak didapatkan orang lain. privilige bersatu bersama keluarga. banyak sekali orang orang yang meninggalkan privilege. begitu kata orang. padahal mereka sudah memilih dan bertanggungjawab dengan konsekuensinya. kita orang kadang memandang sebelah mata. bisa jadi meninggalkan satu privilige tetapi mendapatkan privilige lain yg orang gabisa. dan kadang kita cukup sempit melihat satu privilege semata. di kantor saya contohnya banyak yang menganggap, di kantor pusat, di homebase, di kpp bergengsi, jadi ajudan, dekat bos adalah privilege utama. ya memang benar. tapi jan menutup diri dengan kemungkinan privilege lain. dekat dengan kekuasaan tapi bertemu seminggu sekali dengan keluarga di kota kecil jauh dari mana-mana tapi bersama keluarga tercinta anda pilih mana? semuanya ada privilegenya."
]

thread2 = [
    "Yang kenal saya tau saya paling gemes saat orang anggap Kartini tokoh emansipasi perempuan. Kalau baca surat2nya, jelas Kartini jauh lebih besar daripada sekedar tokoh perempuan. Kartini adalah salah satu pelopor enlightenment di Indonesia UTAS. Biasa, mulai dari background. Saya menemukan Kartini saat sedang galau di New York: saat bosan kerjain disertasi saya browsing buku2 di perpustakaan dan nemu buku Kartini. Saya baca2...ternyata wow banget isinya. Kartini termasuk generasi orang Indonesia yang menjadi aware dengan perubahan zaman. Sadar butuh transformasi identitas karena masa lalu yang diwariskan sudah tak memadai. Tapi juga idealisme barat tidak terasa cocok sepenuhnya. Rasanya seperti kejepit diantara 2 dunia :D. Saat itu saya sudah lebih 5 tahun di NYC dan mulai krisis identitas antara identitas Indonesia & Amerika, dan antara ilmuwan yg skeptis & ""pejuang"" yg percaya. Kartini bergelut dengan masalah yang sama, 100 tahun sebelum saya. What I mean by a liberal education is firstly to nurture Javanese people to become true Javanese, awakening within them a deep love for their nation and country with eyes and heart open to see beauty and to recognize their own needs. Yang saya suka, Kartini sadar ini bukan sebuah pilihan antara timur&barat, atau modern&tradisional. Tapi proses transformasi melalui pendidikan dan kebebasan berpikir. Kartini menulis: We do not intend to make our students to become half Europeans or Javanese mimicking Europeans. Kartini adalah contoh bagaimana interaksi lintas budaya akan memperkaya pemikiran: By crossing plants and animals from many different species we obtain a new better breed of plants and animals. Is it not that way with cultures of different nations?. We would like to share with them all the good things of European culture, not to replace or erase the beauty of their own culture but rather to bring out the radiance of that culture. Kartini dianggap anomali baik oleh kolonial Belanda maupun bangsawan Jawa. Terakhir, ini ocehan saya tiga tahun lalu tentang Kartini 😎 https://t.co/pLVzES266k. If we combine what is good from one nation with something good of another, will not this mixture grow into a culture that is even more noble?. Jadi Kartini bukan sekedar tokoh emansipasi perempuan. Kartini adalah Ibu Intelektual modern Indonesia🤓. SELESAI... Btw, surat2 nya Kartini bisa dibaca disini https://t.co/V7KIZlMHwI."
]

thread3 = [
    "Lagi ke dokter gigi dan denger pasien anak kecil umur 6 tahunan teriak2 dan dokternya bujuk bujukin buka mulut. Terus mikir kenapa anak2 saya gak pernah ada drama ya kalau ke dokter gigi waktu mereka kecil? Oohh ya terus teringat apa yg saya lakukan... Utas. Jadi dari anak2 umur sekitar setahun dua tahun, setiap saya ke dokter gigi, anak2 saya ajak. Saya ijin ke dokter saya apakah anak2 boleh masuk utk lihat supaya mereka kenal. Dan dokter gigi saya malah senang. Mereka duduk dan lihat ibunya diperiksa, ditambal atau scalling. Mereka lihat kursi dokter gigi dengan lampu besar biasanya saya jelaskan ini kursinya, bagus ya ada lampunya besar. Ketika sudah selesai saya jelaskan yg dilakukan dokter gigi kepada saya. Untung dokter giginya juga ramah. Kadang anak saya suka dikasih mainan. Nah setelah sudah sudah sekali dua kali ikut, mereka pemeriksaan pertama kali, ketika gigi mereka sehat, bukan ketika gigi mereka sakit. Mereka diperiksa, dilihat2 doang sama dokternya. Lalu dikasih tahu cara sikat gigi yang benar. Pulangnya dikasih mainan 😁. Akhirnya anak anak malah suka ke dokter gigi, kalau udah 6 bln gak ke dokter gigi suka ingetin saya. Ketika kakak usia SMP mulai pasang kawat gigi udah gak takut sama sekali. Saya temenin awal awal, habis itu kalau kontrol bulanan dia sudah berani sendirian, saya tunggu di luar. Anak saya yg kecil beberapa bulan lalu juga cabut beberapa gigi susu yg sudah goyang agar gigi dewasanya bisa keluar. Ya biasa saja tanpa drama. Karena sudah biasa ke dokter gigi sebelumnya nganterin saya atau ikut nganter kakaknya pasang kawat. Begitu mungkin tips tipsnya, jadikan kunjungan ke dokter gigi sebagai kunjungan rutin sedinj mungkin dan bukan hanya ketika mereka sakit gigi. Kasih hadiah kalau mereka misalnya cabut gigi atau tambal gigi. Hapusan, rautan, jepit rambut, pinsil warna, mainan, jangan permen 😁. Biasakan gosok gigi terutama malam sebelum tidur wajib banget. Anak2 saya sejak kecil gak bisa tidur kalau belum sikat gigi akhirnya. Saya kalau udah capek banget dan ketiduran belum sikat gigi dibangunin sama anak saya disuruh sikat gigi dulu 😂. Oh iya sekalian ngasih tahu deh, saya tadi ke Rumah Sakit Gigi dan Mulut (RSGM) Unpad di Sekeloa, Bandung. Tempatnya enak, buka terus walau libur dan sangat terjangkau. Minggu lalu bedah mulut minor dan hari ini cabut jahitan. Boleh ke sini ya kalau sakit gigi, recommended dah  https://t.co/MNkaJG0h3s. Menjawab beberapa pertanyaan jadi kalau ke RSGM tidak perlu swap antigen. Hanya diukur suhu lalu ada assessment covid dulu. Biayanya kalau di RSGM murah lah pokoknya. Pendaftaran awal 25 ribu bikin kartu pasien, tadi saya cabut jahitan cuma 60 ribu sama dokter residen. lanjut... Rontgen gigi panoramix harganya 140 ribu di RSGM. Anak saya scalling atas bawah biayanya 300 ribu aja. Yg shock, minggu saya bedah minor (dgn dokter residen bukan spesialis) cabut 1 gigi plus obat 3 macam utk 5 hari termasuk antibiotik hanya 273 ribu 😳😳. Satu lagi nih... Ke dokter gigi memang seharusnya ketika gigi tidak sakit. Kalau giginya sdh sakit, harus minum obat dulu beberapa hari, baru bisa ditambal. Yuk ke dokter gigi sebelum sakit 6 bln sekali, supaya kalau ada yg bolong2 bisa ditambal sebelum gigi jadi sakit. Gigi harus disayang sayang. Gigi permanen selama masih bisa dipertahankan jangan sampai dicabut. Bisa perawatan akar dulu dan tambal. Jadi jangan ikuti dokter gigi yang ngasih saran dicabut aja padahal gigi permanennya masih bagus. Cari dokter lain yg ahli konservasi gigi."

]

thread4 = [
    "INDONESIA terlalu MENYEPELEKAN kasus COVID19 pada ANAK!  Situasi Indonesia beda dengan negara lain yang nihil kasus anak meninggal karena COVID19. Ada data yang tak diumumkan pemerintah.  Bikin utas juga nih karena makin was-was dengan ramainya anak berkeliaran di tempat publik.. Ini merupakan rangkuman dari telewicara Ayahbunda dengan ketua IDAI via live IG 20 Mei 2020.  Judul: Anak&amp; Kesehatannya di Masa Pandemi Host: Gracia Danarti (pimred Ayahbunda) Narsum: Dr.dr.Aman B. Pulungan Sp.A(K) [beliau juga pres. dr.Sp.A Asia Pasifik dan ahli diabetic anak). Masih bisa disaksikan di sini:   https://t.co/LlPQpopynD  https://t.co/ymz4QDWfWa. (1) Kita mulai dengan angka-angka pasien anak dg COVID19 di Indonesia PDP 3300-3400 PDP Wafat 129 Konf(+) 584 Konf (+) Wafat 14  Cat: belum utuh karena masih banyak daerah belum bisa menyerahkan data.  #Covid19_anakINA. (2)Sumber data: dihimpun langsung dari dr.Sp.A cabang IDAI se-Indonesia yang merawat pasien anak dg Covid19.  Mengapa dalam statistik resmi, angka pasien anak seolah kecil? Karena anak tidak masuk prioritas tes. Jadi, angka masuk hanya pasien rawat. #Covid19_anakINA. (3) Prioritas tes kita (yg masih sangat minim) adalah pekerja aktif. Karenanya pengetesan lebih banyak dilakukan di tempat kerja, perbelanjaan, stasiun, bandara, dll.  Padahal angka pasien anak dg #COVID19 INA TERTINGGI se-Asia. #Covid29_anakINA. (4) Latar belakang status kesehatan #anak Indonesia:  Tanpa pandemi ini, dr.Sp.A masih bergulat dengan banyaknya kasus infeksi pada anak.  ""Pembunuh #anak Indonesia"" tertinggi: -TBC -Diare (rangking bergantian tiap tahun) #COVIDー19 --&gt; ancaman infeksi baru. BISA KENA &amp; FATAL. Gejala #COVID19 pada anak: ⚫ saluran pernapasan: panas, batuk, sesak ➡️pneumonia  ⚫ saluran cerna: mual, muntah➡️diare  ⚫ tanpa gejala  Konsisten dg masalah infeksi anak di INA. #Covid19_anakINA. (6) Konsekuensi penularan: Anak bukan hanya bisa menularkan #VirusCorona via DROPLET, tapi juga FESES.  #Covid19_anakINA. (8) BAGAIMANA ORTU jaga anak saat pandemi? ➡ #dirumahaja  ➡ Disiplin protokol kebersihan ketat: -sering cuci tangan -pulang dari tempat publik mandi dulu dan bersih-bersih baru berinteraksi dg anak. ➡ Perhatikan 3 hal pada anak (cont.) #Covid19_anakINA. (9) 3 HAL yang harus ortu pantau pada anak: 1. Pertumbuhan 2. Perkembangan 3. Imunisasi  #Covid19_anakINA. (10) TEKNISNYA ⚫ Ukur tinggi dan massa anak (yah, BB deh). Bisa diplot di aplikasi dari #IDAI ""primaku"". ⚫ Perhatikan nutrisi anak ⚫ IMUNISASI JANGAN PUTUS  #Covid19_anakINA. (11) BAHAYA kalau imunisasi sampai putus. Pandemi ini belum selesai nanti bisa disusul wabah lain: pertusis, polio, dll.  Cari klinik/RS yang hanya melayani imunisasi, dan tidak menerima anak sakit. #Covid19_anakINA. Para ortu, yuk jaga anaknya dengan baik. Optimalkan sistem komunitas dg komunikasi intens via gawai, jgn rapat RT di pos ronda, PLEASE #COVIDー19. Tag mas @joeyakarta deh, jogja rame banget. Banyak ortu butuh diingatkan. Demi kebaikan bersama, nggih. Srmoga kita semua bisa selamat dan tetap sehat. #JanganKeluarRumah  #janganmudik  #COVID19 #Covid19_anakINA. Sumber bacaan: ⚫ FAQ ttg Covid19 dari IDAI  https://t.co/pypEAMivUG  ⚫ IDAI: Tidak benar anak tidak rentan terhadap Covid19  https://t.co/iM9GMmCxnt."
]

thread5 = [
    "Mau jualan, tapi bingung jualan apa? nih jawabannya! ------- 5 Cara Dapetin Ide Bisnis yang Pasti Laku [Sebuah Utas] ------- silakan klik LIKE atau Bookmark untuk dibaca nanti  https://t.co/WCyesMv3RO. Pertama, temukan masalah.  kamu bisa membuat list hal apa saja yang ""menyebalkan"", ""menyakitkan"" atau ""merepotkan"" di kegiatan sehari-harimu. lalu, coba buatlah solusi untuk itu. karena hasil polling di Channel Telegram  https://t.co/zZC0czzI86 kemarin mayoritas merasa bingung mau jualan apa, maka utas ini dibuat untuk menjawab kebutuhan tersebut. silakan follow akun ini kalau kamu ingin belajar lebih banyak tentang Bisnis Online, Branding, Niche, CRM. Misalnya, kamu suka banget sama bahan masker yg kamu beli online, sayangnya ukurannya ga pas di wajahmu karena sellernya hanya jual 1 ukuran aja. nah, kamu bisa tuh jualan masker dengan bahan yang sama dan kamu buat ada ukuran S, M, L. Voila! dapet tuh ide bisnis baru!. Ketiga, bantu Komunitas tertentu. coba pilih satu komunitas yang kamu udah familiar atau bahkan kamu sudah tergabung di sana. lakukan survey kecil, tanyakan masalah apa yang paling sering mereka alami. nah, kamu bisa membantu memberikan solusi bagi mereka!. Keempat, lihat passionmu! coba tanyakan ke diri kamu sendiri, hal apa yang bersedia kamu lakuin untuk orang lain walaupun KAMU TIDAK DIBAYAR. kenapa kok ada syarat tidak dibayar? supaya kamu benar2 dapetin tuh apa yg emang beneran kamu suka dan bisa berikan performa yg terbaik. Kelima, Trend is Your Friend! yup, jeli sama trend yang ada dan manfaatin tuh hal-hal viral yang lagi naik banget sekarang apa aja. ini bisa pakai berbagai macam cara ya, bisa kamu lihat dari konten2 yang viral, games yg viral, artis kpop, pakai google trends juga bisa. buat yang mau join Channel Telegram buat ngobrol bisnis kuyyyy join  https://t.co/zZC0czzI86."
]

thread6 = [
    "Kenapa klub-klub elite Eropa ngotot membentuk Super League? Apa uang yang selama ini mereka dapatkan tidak cukup? Emangnya berapa banyak yang akan mereka dapat via Super League?  Ini Penjelasan Super League dari sudut pandang ekonomi klub-klub itu. Baca lengkapnya di utas ini! 👇  https://t.co/jWAxRDoTV1. Berdasarkan @Swissramble, para peserta Super League ini rugi 1,2 miliar poundsterling pada musim 2019/20. Itu, tentu saja, adalah impak dari pandemi.   Perlu dicatat juga, musim lalu cuma 3 bulan terakhir aja yang ada pengaruh dari situasi pandemi.  https://t.co/YuUirP9GRn. Bayangin musim ini (yang full main di situasi pandemi), seberapa banyak lagi kerugiannya? Pasti lebih besar lagi kan.  Klub-klub sama sekali tak dapat pemasukan dari hari pertandingan (tiket, merchandise, FnB selama laga, dll.). Duit berkurang lagi. Padahal pemasukan dari hari pertandingan itu angkanya nggak sedikit lho. Liverpool, misalnya, bisa dapetin 84 juta pounds di musim normal hanya dari pemasukan match day aja.  Duit segitu hilang kan lumayan banget berasanya. Dari kerugian di musim 2019/20 aja klub-klub udah kesulitan belanja pemain. Real Madrid, misalnya, harus nggak belanja karena mereka juga kudu renovasi stadion.  Barcelona harus jual pemain dulu sebelum beli. Sementara Liverpool mengedepankan sistem ""DP dulu, bayar belakangan"". Keterbatasan belanja pada akhirnya berpengaruh pada performa klub juga. Tentu itu membuat para pemilik klub jadi gelisah.  Sejak akhir 2020 lalu, ide-ide liar untuk mencari corong uang lain pun muncul. Mulai dari Project Big Picture, European Premier League, sampai Super League. Awalnya ide datang dari dua taipan Amerika Serikat, Joel Glazer (United) dan John W Henry (Liverpool).  Dan ketika ide itu bergulir ke owner klub-klub lain, semua sepakat dan mulai keranjingan ide bikin turnamen baru ini. Lalu muncullah Super League. Cikal bakal turnamen ini sebenarnya sudah tercium sejak Desember saat Florentino Perez, Presiden Madrid sekaligus Chairman Super League, bilang gini di salah satu pertemuan klub.  ""Kita harus meningkatkan kualitas kompetisi dan aspek kompetitif turnamen yang kita ikuti."". Andrea Agnelli kemudian ikut-ikutan: ""Saya sangat menghormati semua yang dilakukan Atalanta, tetapi, tanpa sejarah internasional dan berkat hanya 1 musim yang hebat, mereka memiliki akses langsung ke Liga Champions. Benar atau tidak?""  Dia merasa yang layak hanya klub besar saja. Di awal tahun ini, Perez dan Agnelli kemudian mengadakan pertemuan. Proyek ini hanyalah keserakahan dari para owner klub.  Selama ada kesempatan buat dapetin uang lebih, ya, hajar. Nggak peduli tradisi dan ikatan yang udah mereka jalani dengan kompetisi yang ada sekarang. Culasnya, keputusan ikut serta ini seperti tak didiskusikan dulu dengan pihak lain seperti suporter atau asosiasi pemain (FIFpro).  Yang jelas, inti dari ini adalah soal cuan. Apa tagar #AgainstModernFootball sudah muncul?. Menarik juga menantikan suara dari para pemain dan pelatih/manajer. Ole Solskjaer semalam kedapatan ditanya soal ini sama wartawan, tapi dia menolak jawab.  Juergen Klopp, sejak tahun lalu, sudah menolak ide Super League ini. Sumber utas kami: - The Athletic - The New York Times - Bloomberg - @SwissRamble  - Pers Rilis Super League.  Utas disusun oleh @bergasss.  https://t.co/UYG8DIpGtB."
]

thread7 = [
    "Kenapa ‘cowo cowo’ mundur?  [sebuah utas]  https://t.co/AL3VP7uBlW. Dari hasil polling kemaren kayanya banyak yg penasaran sama thread ini, karna mumpung emang akunya pengen bahas. Apa aja sih alasan cowo2 mundur atau bahasa kerennya ‘ghosting’ secara tbtb pas masa pdkt?. Biasanya kami (cowo) masih menilai (cewe) yang bakal dipacarin mulai dari ngedate pertama, ke2, ke3 hingga ngedate seterusnya. Di saat kalian para cewe2 udah ngerasa nyaman dan yakin bakal jadi pacar kami para cowo2, ada kemungkinan kami berubah pikiran trus mendadak ngilang tanpa kabar dan jejak. Jika kalian para cewe pernah ngalaminnya, atau gak mau mengalaminnya, ini ada beberapa alasan yang menjadi penyebabnya!. 1. Kalian (cewe) terlalu ribet.  Kebanyakan kami (cowo) emang suka tantangan. namun jika kalian para cewe terlalu pasif, jaim, bahkan sok jual mahal, rasanya wajar banget kalo kami bakal mengundurkan diri.  https://t.co/7rZnIO0vjN. Hanya ada dua jenis cowo yang bersedia capek meladeni keribetan kalian para cewe;  Cowo polos nggak laku yang rela berjuang keras karena tidak ada pilihan lain. Dan cowo fuckboy penjahat-kelamin yang memang selalu kecanduan tantangan. 2. Kalian (cewe) terlalu high maintenance, high sociality, dan jenis2 high lainnya.  Dan kami (cowo) khawatir nantinya kami yang turut ikut menanggung biaya ’pergaulan duniawi’ kalian para cewe ketika pas udah jadian.  https://t.co/DqlNI7fABN. A different cases kalo kalian cewe2 ngelakuin hal tadi berdasarkan keinginan sendiri dan make dana pribadi tanpa harus kami ikut campur.  “Diluar dari konteks pelit”. 3. Belum begitu yakin sama kalian (cewe).  Pas udah saling mengenal satu sama lain kami (cowo) menemukan hal2 dari kalian yang bertentangan. Mulai dari soal prinsip atau dari sifat &amp; sikap kalian para cewe yang gak sesuai sama kriteria.  https://t.co/0XjtEpf9zK. Nahh, point diatas adalah beberapa contoh atau mungkin template yang di jadikan alasan atas gagalnya memulai hubungan.   Pesan moral yg bisa diambil dari thread ini jadikan reminder buat kalian para cewe supaya jangan baper duluan saat lagi pdkt. Yaa mungkin thread ini gak begitu valid dan sensitive untuk beberapa kalangan setidaknya relateable untuk sebagian orang. 😉. @__LEICA__ Nah, cowo juga demikian.. kami menghilang karena sadar bahwa kalian mungkin semenarik, sememikat, dan semenyenangkan itu. Simpel."

]

thread8 = [
    "Kami mewawancarai sejarawan Minangkabau, sejarawan kuliner, guru besar ilmu gizi, dan 100 RM Padang untuk artikel ini: Semua yang harus kamu tahu tentang Nasi Padang.   Sebuah utas.  https://t.co/Fp6ggr1e3N  https://t.co/mo9Kf9DDj9. Kata ""Nasi Padang"", atau ""Rumah Makan Padang"" baru mulai populer pada akhir 1960-an, sebagai penamaan kontemporer bagi restoran Minangkabau. Sekarang, Nasi Padang ada di mana-mana, sampai di seluruh dunia. Lokasi RM Padang terjauh dari Padang ada di...  https://t.co/KlxN0Jasas. Yang jelas, survei BBC menemukan bahwa 79 persen warung Padang memberikan nasi lebih banyak untuk pesanan yang dibungkus.  https://t.co/cXpwlgF1eg  https://t.co/Ww8Azx5NLs. Menurut Guru Besar Ilmu Gizi Universitas Andalas, makan gorengan jauh lebih berbahaya dari makanan yang bersantan.   Selain itu, makanan Minang kaya akan bumbu seperti jahe, kunyit, lengkuas, dan daun serai yang berperan positif untuk tubuh. Bukan berarti boleh rakus ya.   Kami membuat kalkulator kalori untuk membantu menghitung berapa kalori yang terkandung pada sepiring Nasi Padang kamu.  https://t.co/cXpwlgF1eg  https://t.co/odyOrcQzKf. Misalnya, ternyata sepotong telur dadar berkalori jauh lebih tinggi daripada rendang.   Kalori terendah? Daun singkong 😅  https://t.co/cXpwlgF1eg  https://t.co/lVtPrfUPDD. Soal keaslian, sebagian besar warung Padang yang kami survei di Jabodetabek ternyata masih dimiliki oleh orang Minang. Sebagian masih pakai sistem bagi hasil.  https://t.co/yO4pe8W7rI. Baca artikel selengkapnya di tautan ini ya  https://t.co/cXpwlgF1eg  https://t.co/GnD52hoqqy."
]

thread9 = [
    "☣️ TOXIC RELATIONSHIP ☣️ - sebuah utas hubungan yang merusak, namun memabukkan -  #KhotbahDukun  https://t.co/GOVgzTegAM. Pernahkah kamu berada dalam suatu hubungan... dan berbagai perasaan negative bermunculan?  • Masalah muncul tak kunjung usai  • Pertengkaran berujung frustrasi  • Rasa bersalah yang berlebihan  ..bisa jadi itu ciri-ciri hubungan yang bermasalah.. Alih2 tumbuh bersama pasangan..  Individu di dalam hubungan yang toxic malah akan saling menghambat, bahkan merusak  https://t.co/O9aAY6JLDh. Kenapa hal ini bisa terjadi?  Tentu ada pihak di hubungan yang berusaha untuk memenuhi keinginannya sendiri..  ..dan pihak tersebut akan menggunakan hubungan sebagai ""alat"" untuk memuaskan ego pribadinya.. ""Pihak"" ini bisa salah satu atau keduanya yak..  https://t.co/G4h6sUFJTY. Ego tersebut dapat berbentuk berbagai kebutuhan..  Kebutuhan untuk selalu diturutin.. Kebutuhan untuk selalu dihormatin.. Kebutuhan untuk selalu diutamakan..  Macem2 lah, intinya isu psikologis yang seharusnya diselesaikan scr mandiri.. Nah, ujung2nya.. pasangan tersebut akan saling manipulasi..  ""Kalau kamu ngga prioritasin aku, aku ngga mau lagi bla bla..""  ""Awas kalau kayak gini lagi, aku bakal..""  ""Lain kali, aku ngga mau lagi kalau kamu masih..""  Syarat demi syarat dititahkan..  https://t.co/7RYPBukmxu. Ujung2nya apa?  Saling manipulatif, Saling tergantung, Saling merasa bersalah..  Lengkap udah. Perlu dipahami, bahwa memiliki pasangan bukan berarti kamu bisa bebas dari tanggung jawab pribadimu..  Perasaanmu ya urusanmu, bukannya mengharapkan pasangan yang menjaga keseimbangannya~  https://t.co/tluD3qcOI4. Kenapa?  Karena hubungan itu tidak dibentuk dari 2 individu yang saling tergantung..  Melainkan dari 2 individu dewasa yang ingin tumbuh bersama..  https://t.co/SNbahadKI8. Ambil waktumu, untuk melihat perkembangan dirimu sendiri..  Jangan melulu menuntut pasangan sebelum kamu mau berubah . .  Whatever you do for yourselves, it will benefit you in the future..  Doesn't matter if it's with or without him/her :)  ☣️The End☣️  https://t.co/QOINLO8jFT. Final notes :  Thread ini juga bahan refleksi, Jangan dipake untuk nuntut atau nyalahin pasanganmu 🤪."
]

thread10 = [
    "Mau ngasih tau sama antis yg so tau bilang.... ""EXO DARI AWAL UDAH BESAR, KARENA MEREKA DARI AGENSI BESAR""   B A C O T ! !   -sebuah utas- Before.                 After  https://t.co/tpXomsIaDW. Dan kalian tau EXO saat itu tampil dibayar bukannya memakai uang melainkan dibayar dgn SEKANTONG BERAS!! tapi mereka tdk marah, mereka seneng, bersyukur, bahagia walaupun gk seberapa mereaka dibayar (syg byk2 sama mereka❤)  https://t.co/k20uyHvjHa. Berapa lama yg EXO lalui untuk mendapatkan kemenangan diacara musik pertama mereka?  Dimana sekarang ini grup baru hanya membutuhkan beberapa hari untuk mendapatkan kemenangan, tapi lain dengan EXO mereka membutuhkan waktu 400 hari lebih untuk menang diacara musik. (anjirrr😭). Dan beberapa hari setelah itu EXO melakukan interview diradio dan sehun bilang  ""semuanya mari saling bersandar jgn pernah bertengkar, aku berharap kita selalu bersama sampai akhir dan bahagia, aku sangat mencintai kalian"" (menangis tumben nih anak ayam bnr😭😭)  Pic. Pemanis  https://t.co/2lJiGK4g8z. Setelah itu overdose menang diacara penghargaan musik, dan kalian tau? itu tidak berjalan dgn lancar, hanya suho leader kita yg datang saat itu😭 karena member lain masih syok dgn apa yg mereka terima:( tapi suho? dia naik ke atas panggung sendirian...  https://t.co/KtcEIBuosg. menerima penghargaan sendirian, menyanyi sendirian, berpidato sendirian, menari sendirian:( suho tidak meninggalkan tanggung jawab nya sbg leader, tidak salah dia dijuluki ""THE BEST OF LEADER"". ❤  https://t.co/giAtsoRShN. ""Quaruple million seller"" EXO juga merupakan satu2nya idol yg mampu mendapatkan daesang mama selama 5tahun berturut2 hingga nama mereka tercatat di ""guiness book of world records 2018"".  https://t.co/OnvbAa5dh1. Tiket habis dalam 0.5 detik ribuan orang datang ke fansgin dan menghadiri konser mereka, EXO memenangkan 200lebih penghargaan salah satunya penghargaan paling sepesial yaitu ""prime minister award"". penghargaan paling tinggi karena diberikan olh pemerintahan korea (terharu😢)  https://t.co/k9xpkafPcD. Jadi tolong jgn asal menyimpulkan EXO seperti apa, mereka juga berjuang ga nikmatin langsung dr awal, mereka dari bawah walaupun dr agensi besar, setiap grup pnya ceritanya masing2. ga semuanya mulus pasti ada likalikunya juga, diem lebih baik daripada so tau☺."
]

# ========== 2. Remove URL, emoji, and mention. Modify #hashtag = word, and HTML entities (done) ==========
'''Preprocessing by removing unused characters.'''
# replace new line with period symbol
def replaceNewLine(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r'(?<!\.)\n\n(?!\.)', '.', all_text[i])

# remove HTML entity
def replaceHTMLentity(all_text):
    h = html.parser
    for i in range(len(all_text)):
        all_text[i] = h.unescape(all_text[i].replace('\n',''))

# remove URL
def removeURL(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r'http\S+', '', all_text[i])

# remove mention
def removeMention(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub('@[^\s]+ ', '', all_text[i])

# remove hashtag
def replaceHashtag(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r'#([^\s]+)', r'\1', all_text[i])

def replaceDoubleQuotation(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r'""', '"', all_text[i])

# remove emoticon - example: :) | :( |:D | :3 | :* | xD | :'(
with open('documents/Emoticon_Dict.p', 'rb') as fp:
    Emoticon_Dict = pickle.load(fp)
def removeEmoticon(all_text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in Emoticon_Dict) + u')')
    for i in range(len(all_text)):
        all_text[i] = emoticon_pattern.sub(r'', all_text[i])

# remove emoji
def removeEmoji(all_text):
    for i in range(len(all_text)):
        all_text[i] = emoji.replace_emoji(all_text[i], replace='')
        # emoji.get_emoji_regexp().sub(r'', text.decode('utf8'))

# remove repeated exclamation mark
def removeRepeatedExclamation(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r"(\!)\1+", '!', all_text[i])

# remove repeated question mark
def removeRepeatedQuestion(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r"(\?)\1+", '?', all_text[i])

# remove repeated period
def removeRepeatedPeriod(all_text):
    for i in range(len(all_text)):
        all_text[i] = re.sub(r"(\.)\1+", '.', all_text[i])

# ========== 3. Give period to tweets without period at end of sentence to define a sentence (done) ==========
'''Add period at every end of Tweet if it is a sentence without a continuation symbol'''
def addPeriod(all_text):
    for i in range(len(all_text)):
        if not all_text[i].endswith(('.', ',', '-', '?', '!', '~')):
            all_text[i] += '.'


'''Create text_cleaning function to combine all text cleaning steps'''
def text_cleaning(all_text):
    replaceNewLine(all_text)
    replaceHTMLentity(all_text)
    removeURL(all_text)
    removeMention(all_text)
    replaceHashtag(all_text)
    replaceDoubleQuotation(all_text)
    removeEmoji(all_text)
    removeEmoticon(all_text)
    removeRepeatedExclamation(all_text)
    removeRepeatedQuestion(all_text)
    removeRepeatedPeriod(all_text)
    addPeriod(all_text)
    return all_text

# ========== 4. Concatenate all strings in list ==========
'''Concatenate all Tweets into one string'''
def concatenate(all_text):
    text_concatenated = ' '.join(all_text)
    return text_concatenated


# ========== 5. Final text cleaning for spacing and symbol problems ===========
def clean_after_concat(text_concatenated):
    '''Remove double whitespaces'''
    text_cleanfinal = ' '.join(text_concatenated.split())
    '''Add space for no whitespace'''
    text_cleanfinal1 = re.sub(r'(?<=[!?.,])(?=[^\s])', ' ', text_cleanfinal)
    '''Remove whitespace before symbols'''
    text_cleanfinal2 = re.sub(r'\s+([?.!,])', r'\1', text_cleanfinal1)
    '''Remove period symbols after/before other symbols (e.g. "?.", ".!", "-.", and others)'''
    text_cleanfinal3 = re.sub(r'[?!.][?.!]+', lambda x: x.group()[0], text_cleanfinal2)
    return text_cleanfinal3


# OPTIONAL!
# ========== 5. Named-Entity Recognition to avoid text normalization for important names ==========
'''NER to avoid text normalization for important names of people, organization, and other things that can be named'''


# ========== 6. Text normalization ==========
'''Text normalization by replacing colloquial lexicon with its equivalent formal word to reduce noise in the data'''
# using indoNLP library
def normalize(text_cleanfinal3):
    text_normalized = replace_word_elongation(replace_slang(text_cleanfinal3))
    return text_normalized

# slang search
def slang_search(text_cleanfinal3):
    pattern = re.compile(rf"(?i)\b({'|'.join(SLANG_DATA.keys())})\b")
    matches = re.findall(pattern, text_cleanfinal3)
    return matches

def clean_until_concat(thread):
    clean_thread = text_cleaning(thread)
    concat_thread = concatenate(clean_thread)
    clean_concat_thread = clean_after_concat(concat_thread)
    return clean_concat_thread

def clean_until_normalize(thread):
    clean_thread = text_cleaning(thread)
    concat_thread = concatenate(clean_thread)
    clean_concat_thread = clean_after_concat(concat_thread)
    normalized_thread = normalize(clean_concat_thread)
    return normalized_thread

# INDONESIAN TEXT NORMALIZATION MANUALLY
# def normalisasi(tweet):
#     kamus_slangword = eval(open("slang_indonesia.txt").read()) # Membuka dictionary slangword
#     pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
#     content = []
#     for kata in tweet:
#         filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
#         content.append(filteredSlang.lower())
#     tweet = content
#     return tweet
# df['Normalization'] = df['Stopword_Removal'].apply(lambda x: normalisasi(x))
# df.head(10)

# with open('documents/abbrev.csv') as file:
#     slang_map = dict(map(str.strip, line.partition('\t')[::2])
#     for line in file if line.strip())

# slang_words = sorted(slang_map, key=len, reverse=True)
# regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
# replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

# slangDict = pd.read_csv('documents/abbrev.csv', index_col = 1 , header=0).to_dict()

# def replaceSlang(text):
#     words = text.split()
#     new_words = []
#     for word in words:
#         if word.lower() in slangDict:
#             word = slangDict[word.lower()]
#         new_words.append(word)
#     new_text = " ".join(new_words)
#     return new_text


# ========== 7. Calculate relevance of each sentence and rank it based on importance | 8. Remove irrelevant/out-of-topic sentence based on the rank and some common vocabularies ==========
'''Rank each sentence importance using semantic search'''
def semantic_search(text_normalized):
    text_with_semantic_search = do_semantic_search(text_normalized)
    return text_with_semantic_search


def preprocessing(thread):
    clean_thread = text_cleaning(thread)
    concat_thread = concatenate(clean_thread)
    clean_concat_thread = clean_after_concat(concat_thread)
    normalized_thread = normalize(clean_concat_thread)
    semantic_search_thread, sentence_removed = semantic_search(normalized_thread)
    return semantic_search_thread, sentence_removed

# print(preprocessing(thread8))

# processed_text, sentence_removed = preprocessing(thread8)
# print(processed_text)

# 8. Summarize the text using a language model
