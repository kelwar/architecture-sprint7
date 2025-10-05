### ./build_index.py

Использовалась модель эмбеддингов all-MiniLM-L6-v2 + FAISS.

База знаний сформирована на основе малой части статей энциклопедии мира ПЛиО (вселенная по серии книг "Песнь льда и пламени" Джорджа Мартина).
Чтобы терминов не было слишком много, часть подробного материала из статей была намеренно убрана.
В задаче был сделан фокус на общие географические и геополитические сведения о локациях и домах Вестероса.
Все имена собственные из статей заменены на основании справочника terms_map.json, заголовки и пустые строки убраны.

Количество чанков для обработки: 888
Время создания индекса: 4.67 с.
Запрос:  Принцесса Песчаной страны
Время поиска релевантного чанка: 0.04 с.
Relevant chunks:
1 page_content='Принцесса Белль Редсан – старшая дочь правящего принца Пустынной страны, главы дома Редсанов Гута.' metadata={'source': './knowledge_base/redsun_belle.txt'}
2 page_content='принцесса Белль Редсан — их старшая дочь и наследница.' metadata={'source': './knowledge_base/redsuns.txt'}
3 page_content='принцесса Пустынной страны. Является одним из ПОВов серии[1].' metadata={'source': './knowledge_base/redsun_belle.txt'}
4 page_content='интриган. Принц Пустынной страны испытывает глубокую привязанность к своей сестре, а также имел' metadata={'source': './knowledge_base/redsun_gut.txt'}
5 page_content='дочерьми принца Редснейка Редсана.' metadata={'source': './knowledge_base/redsun_belle.txt'}
6 page_content='и «принцессы».' metadata={'source': './knowledge_base/redsuns.txt'}
7 page_content='Принц Редснейк Редсан — принц, младший брат правителя Пустынной страны Гута Редсана. Редснейк не был' metadata={'source': './knowledge_base/redsun redsnake.txt'}
8 page_content='Принц Гут Редсан — принц Пустынной страны и глава дома Редсанов, старший брат Редснейка Редсана. У' metadata={'source': './knowledge_base/redsun_gut.txt'}
9 page_content='Планта, сира Риттера Планта и сира Спаркли Планта.' metadata={'source': './knowledge_base/plant_bella.txt'}
10 page_content='стране дочери Редснейка.' metadata={'source': './knowledge_base/redsun redsnake.txt'}

Запрос:  Старший брат Кроуси
Время поиска релевантного чанка: 0.02 с.
Relevant chunks:
1 page_content='Кроуси — младший брат Маршала Октопуса и старший — Айрона и Эйрона. Стинкеру и Вумен он приходится' metadata={'source': './knowledge_base/octopus_crowsee.txt'}
2 page_content='Маршал — старший брат Кроуси, Айрона и Эйрона. У него было двое детей: Вумен и Стинкер. Его старшие' metadata={'source': './knowledge_base/octopus_marshall.txt'}
3 page_content='брат и ровесник Янг. Описание внешности братьев в оригинале можно понимать и так, что у Шона более' metadata={'source': './knowledge_base/winter_shon.txt'}
4 page_content='Бухта регента: крупнейший город в стране и столица государства, построенная Вильгельмом Тридрейком' metadata={'source': './knowledge_base/zibenlands.txt'}
5 page_content='Айрон Октопус — капитан флота. Младший брат Маршала и Кроуси, и старший Эйрона, а также дядя' metadata={'source': './knowledge_base/octopus_iron.txt'}
6 page_content='родных брата: Янг, Рейвен и Шагги, старшая сестра Бёрд и единокровный брат Шон Винтер. Как и все' metadata={'source': './knowledge_base/greathound_des.txt'}
7 page_content='Сир Спаркли Плант — младший сын Фэта Планта, князя Верхнего огорода, младший брат Смарта и Риттера' metadata={'source': './knowledge_base/plant_sparkly.txt'}
8 page_content='сир Маршал Леон — младший брат Вайса, верховный судья, регент и защитник державы при Киттене Дире.' metadata={'source': './knowledge_base/leons.txt'}
9 page_content='стороны Кроуси[2].' metadata={'source': './knowledge_base/octopus_iron.txt'}
10 page_content='Янг Грейтхаунд — старший сын Патрика Грейтхаунда и Катрин Грейтхаунд, наследник отца. Старший брат' metadata={'source': './knowledge_base/greathound_yang.txt'}