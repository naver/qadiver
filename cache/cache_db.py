# Copyright 2019-present NAVER Corp.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from hashlib import md5
import sqlite3

def get_hash(text):
    return md5(text.encode("utf-8")).hexdigest()

class QACache:
    def __init__(self, path="cache/qa_cache.db", use_memory=True):
        self.conn = sqlite3.connect(path)
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS qa_table (id text primary key, q text, c text, a text, p double);")
        cur.close()

        self.use_memory = use_memory
        self.data = {}
        if self.use_memory:
            full = self.get_data_with_query("select id, a, p from qa_table")
            for item in full:
                self.data[item[0]] = [item[1], item[2]]

    def put_answer_prob(self, id_, q, c, a, p):
        cur = self.conn.cursor()
        cur.execute("insert into qa_table values (?, ?, ?, ?, ?);", (id_, q,c,a,p))
        self.conn.commit()
        cur.close()
        
    def put_answer_batch(self, data_list):
        cur = self.conn.cursor()
        cur.executemany("insert into qa_table values (?, ?, ?, ?, ?);", data_list)
        self.conn.commit()
        cur.close()

    def get_answer_prob(self, id_):
        if self.use_memory and id_ in self.data:
            return self.data[id_]

        cur = self.conn.cursor()
        cur.execute("select a, p from qa_table where id = ?;", (id_,))
        result = cur.fetchone()
        cur.close()
        return result

    def search_ids(self, word, limit):
        cur = self.conn.cursor()
        cur.execute("select id from qa_table where id like ? or q like ? or c like ? limit ?", 
                    (word+"%", "%"+word+"%", "%"+word+"%", limit))
        result = cur.fetchall()
        cur.close()
        return [v[0] for v in result]

    def get_uans_data(self):
        cur = self.conn.cursor()
        cur.execute("select id, p from qa_table;")
        result = cur.fetchall()
        result_dict = {}
        for item in result:
            result_dict[item[0]] = item[1]
        return result_dict

    def get_data_with_query(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        cur.close()
        return result

    def clear_cache(self):
        cur = self.conn.cursor()
        cur.execute("delete from qa_table;")
    

        
