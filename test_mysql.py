import pymysql
import yaml
import sys
import os
import csv
import time
from datetime import datetime
import threading

# 固定输出目录和数据目录为绝对路径，保持与主程序一致
OUTPUT_IMAGES_DIR = r"E:\积水识别项目\demo0625\demo\output_images"
RUNS_OUTPUT_IMAGES_DIR = r"E:\积水识别项目\demo0625\demo\runs\output_images"
DATA_DIR = r"E:\积水识别项目\demo0625\demo\data"


def load_database_config():
    """从YAML文件加载数据库配置"""

    config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config['database']
    except FileNotFoundError:
        print(f"配置文件未找到: {config_path}")
        return None
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return None


def test_mysql_connection():
    """测试MySQL数据库连接"""

    db_config = load_database_config()
    if not db_config:
        print("无法加载数据库配置")
        return False

    connection = None
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor,
        )

        print("数据库连接成功！")
        print(f"连接信息: host={db_config['host']}, port={db_config['port']}, database={db_config['database_name']}")

        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            result = cursor.fetchone()
            if result:
                print(f"MySQL版本: {result['VERSION()']}")
            else:
                print("无法获取MySQL版本信息")

            create_table_sql = """
            CREATE TABLE IF NOT EXISTS test_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            print("测试表创建成功或已存在")

            insert_sql = "INSERT INTO test_table (name) VALUES (%s)"
            cursor.execute(insert_sql, ('测试数据',))
            connection.commit()
            print("测试数据插入成功")

            select_sql = "SELECT * FROM test_table"
            cursor.execute(select_sql)
            results = cursor.fetchall()
            print("查询到的数据:")
            for row in results:
                print(f"ID: {row['id']}, Name: {row['name']}, Created At: {row['created_at']}")

    except pymysql.MySQLError as e:
        print(f"数据库连接失败: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        try:
            if connection:
                connection.close()
                print("数据库连接已关闭")
        except Exception:
            pass

    return True


def create_database_tables():
    """创建数据库表结构"""

    db_config = load_database_config()
    if not db_config:
        print("无法加载数据库配置")
        return False

    connection = None
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor,
        )

        with connection.cursor() as cursor:
            create_trains_table_sql = """
            CREATE TABLE IF NOT EXISTS trains (
                id INT AUTO_INCREMENT PRIMARY KEY,
                train_name VARCHAR(50) NOT NULL UNIQUE,
                total_carriages INT DEFAULT 0,
                accuracy_rate DECIMAL(5,2) DEFAULT 0.00,
                correct_count INT DEFAULT 0,
                participated_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_trains_table_sql)
            print("列车信息表创建成功或已存在")

            create_carriages_table_sql = """
            CREATE TABLE IF NOT EXISTS carriages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                train_id INT NOT NULL,
                carriage_id VARCHAR(10) NOT NULL,
                full_carriage_number VARCHAR(50) NOT NULL,
                recognized_plate VARCHAR(50) DEFAULT '',
                recognition_time VARCHAR(50) DEFAULT '',
                is_special BOOLEAN DEFAULT FALSE,
                water_info VARCHAR(20) DEFAULT '无水',
                water_area_ratio DECIMAL(5,2) DEFAULT 0.00,
                top_image_path TEXT,
                side_image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (train_id) REFERENCES trains(id) ON DELETE CASCADE
            )
            """
            cursor.execute(create_carriages_table_sql)
            print("车厢信息表创建成功或已存在")

            try:
                cursor.execute("CREATE INDEX idx_train_id ON carriages(train_id)")
            except pymysql.MySQLError as e:
                if "Duplicate key name" not in str(e):
                    raise e
                print("索引idx_train_id已存在")

            try:
                cursor.execute("CREATE INDEX idx_train_name ON trains(train_name)")
            except pymysql.MySQLError as e:
                if "Duplicate key name" not in str(e):
                    raise e
                print("索引idx_train_name已存在")

        connection.commit()
        print("所有表结构创建完成")
        return True

    except pymysql.MySQLError as e:
        print(f"数据库操作失败: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        try:
            if connection:
                connection.close()
                print("数据库连接已关闭")
        except Exception:
            pass


def insert_train_data(train_name, total_carriages, accuracy_rate, correct_count, participated_count):
    """插入或更新列车数据，避免重复插入"""

    db_config = load_database_config()
    if not db_config:
        print("无法加载数据库配置")
        return False

    connection = None
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor,
        )

        with connection.cursor() as cursor:
            cursor.execute("SELECT id, updated_at FROM trains WHERE train_name = %s", (train_name,))
            existing_train = cursor.fetchone()

            if existing_train:
                update_train_sql = """
                UPDATE trains SET
                    total_carriages = %s,
                    accuracy_rate = %s,
                    correct_count = %s,
                    participated_count = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE train_name = %s
                """

                cursor.execute(
                    update_train_sql,
                    (
                        total_carriages,
                        accuracy_rate,
                        correct_count,
                        participated_count,
                        train_name,
                    ),
                )
                print(f"列车数据更新成功: {train_name}")
            else:
                insert_train_sql = """
                INSERT INTO trains (train_name, total_carriages, accuracy_rate, correct_count, participated_count)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(
                    insert_train_sql,
                    (train_name, total_carriages, accuracy_rate, correct_count, participated_count),
                )
                print(f"列车数据插入成功: {train_name}")

        connection.commit()
        return True

    except pymysql.MySQLError as e:
        print(f"操作列车数据失败: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    finally:
        try:
            if connection:
                connection.close()
        except Exception:
            pass


def insert_carriage_data(train_name, carriage_data):
    """插入车厢数据，避免重复插入"""

    db_config = load_database_config()
    if not db_config:
        print("无法加载数据库配置")
        return False

    connection = None
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor,
        )

        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM trains WHERE train_name = %s", (train_name,))
            train_result = cursor.fetchone()
            if not train_result:
                print(f"未找到列车: {train_name}")
                return False

            train_id = train_result['id']

            cursor.execute(
                """
                SELECT id, updated_at, full_carriage_number, recognized_plate, water_info, water_area_ratio
                FROM carriages
                WHERE train_id = %s AND carriage_id = %s
                """,
                (train_id, carriage_data['carriage_id']),
            )

            existing_carriage = cursor.fetchone()

            if existing_carriage:
                if (
                    existing_carriage['full_carriage_number']
                    == carriage_data['full_carriage_number']
                    and existing_carriage['recognized_plate']
                    == carriage_data['recognized_plate']
                    and existing_carriage['water_info']
                    == carriage_data['water_info']
                    and float(existing_carriage['water_area_ratio'])
                    == float(carriage_data['water_area_ratio'])
                ):
                    print(
                        f"车厢数据已存在且无变化，跳过: {train_name} - {carriage_data['carriage_id']}"
                    )
                    return True
                else:
                    update_carriage_sql = """
                    UPDATE carriages SET
                        full_carriage_number = %s,
                        recognized_plate = %s,
                        recognition_time = %s,
                        is_special = %s,
                        water_info = %s,
                        water_area_ratio = %s,
                        top_image_path = %s,
                        side_image_path = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE train_id = %s AND carriage_id = %s
                    """

                    cursor.execute(
                        update_carriage_sql,
                        (
                            carriage_data['full_carriage_number'],
                            carriage_data['recognized_plate'],
                            carriage_data['recognition_time'],
                            carriage_data['is_special'],
                            carriage_data['water_info'],
                            carriage_data['water_area_ratio'],
                            carriage_data['top_image_path'],
                            carriage_data['side_image_path'],
                            train_id,
                            carriage_data['carriage_id'],
                        ),
                    )
                    print(f"车厢数据更新成功: {train_name} - {carriage_data['carriage_id']}")
            else:
                insert_carriage_sql = """
                INSERT INTO carriages (train_id, carriage_id, full_carriage_number, recognized_plate,
                                      recognition_time, is_special, water_info, water_area_ratio,
                                      top_image_path, side_image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(
                    insert_carriage_sql,
                    (
                        train_id,
                        carriage_data['carriage_id'],
                        carriage_data['full_carriage_number'],
                        carriage_data['recognized_plate'],
                        carriage_data['recognition_time'],
                        carriage_data['is_special'],
                        carriage_data['water_info'],
                        carriage_data['water_area_ratio'],
                        carriage_data['top_image_path'],
                        carriage_data['side_image_path'],
                    ),
                )
                print(f"车厢数据插入成功: {train_name} - {carriage_data['carriage_id']}")

        connection.commit()
        return True

    except pymysql.MySQLError as e:
        print(f"操作车厢数据失败: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    finally:
        try:
            if connection:
                connection.close()
        except Exception:
            pass


def find_image_for_carriage_with_csv_cached(train_name, carriage_id, output_dir=RUNS_OUTPUT_IMAGES_DIR):
    """根据列车名称和车厢ID查找对应图片和积水信息（直接从CSV读取）"""

    try:
        carriage_num = int(carriage_id)
        image_name = f"frame_{carriage_num}.jpg"

        image_path = os.path.join(output_dir, train_name, image_name)

        individual_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, train_name, f"{train_name}_积水识别结果.csv")
        has_water = False
        water_detected_image_path = None
        water_area_ratio = 0.0

        csv_data = None
        if os.path.exists(individual_csv):
            print(f"优先读取个别积水识别结果文件: {individual_csv}")
            try:
                with open(individual_csv, 'r', encoding='utf-8-sig') as f:
                    csv_data = list(csv.DictReader(f))
            except Exception as e:
                print(f"读取个别CSV文件失败: {str(e)}")
                csv_data = None
        else:
            water_summary_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, "所有子文件夹积水识别结果汇总.csv")
            if os.path.exists(water_summary_csv):
                print(f"回退到汇总文件: {water_summary_csv}")
                try:
                    with open(water_summary_csv, 'r', encoding='utf-8-sig') as f:
                        csv_data = list(csv.DictReader(f))
                except Exception as e:
                    print(f"读取汇总CSV文件失败: {str(e)}")
                    csv_data = None

        if csv_data:
            for row in csv_data:
                original_name_key = '原图片名称'
                water_status_key = '是否有积水'
                detected_path_key = '识别后图片路径'
                water_ratio_key = '积水面积占比(%)'

                if row.get(original_name_key, '').strip() == image_name:
                    has_water = row.get(water_status_key, '').strip() == '是'
                    try:
                        raw_ratio = float(row.get(water_ratio_key, '0.0'))
                        water_area_ratio = raw_ratio * 1.3 if has_water else 0.0
                    except (ValueError, TypeError):
                        water_area_ratio = 0.0

                    if has_water and row.get(detected_path_key, '').strip():
                        water_detected_image_path = row.get(detected_path_key, '').strip()
                        if not os.path.isabs(water_detected_image_path):
                            water_detected_image_path = os.path.abspath(water_detected_image_path)

                    print(
                        f"从CSV中找到匹配记录: {image_name}, 有积水: {has_water}, 面积占比: {water_area_ratio:.2f}%"
                    )
                    break

        return {
            'original_path': image_path if os.path.exists(image_path) else None,
            'water_detected_path': water_detected_image_path,
            'has_water': has_water,
            'image_name': image_name,
            'water_area_ratio': water_area_ratio,
        }
    except ValueError as e:
        print(f"处理车厢ID时出错: {str(e)}")
        return {
            'original_path': None,
            'water_detected_path': None,
            'has_water': False,
            'image_name': None,
            'water_area_ratio': 0.0,
        }
    except Exception as e:
        print(f"查找图片信息时出错: {str(e)}")
        return {
            'original_path': None,
            'water_detected_path': None,
            'has_water': False,
            'image_name': None,
            'water_area_ratio': 0.0,
        }


def find_side_image_by_plate_number_cached(train_name, full_carriage_number):
    """根据完整车号查找匹配的侧面图片和识别结果（仅在同名文件夹中）"""

    try:
        csv_path = os.path.join(OUTPUT_IMAGES_DIR, train_name, "plate_results.csv")

        if not os.path.exists(csv_path):
            return {
                'image_path': None,
                'plate_number': full_carriage_number,
                'timestamp': "未知",
            }

        with open(csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if row['plate_number'].strip().upper() == full_carriage_number.strip().upper():
                    return {
                        'image_path': os.path.join(OUTPUT_IMAGES_DIR, train_name, row['image_name']),
                        'plate_number': row['plate_number'],
                        'timestamp': row['timestamp'],
                    }

            csv_file.seek(0)
            next(reader)

            target_digits = extract_digits(full_carriage_number)
            if len(target_digits) >= 5:
                best_match = None
                max_common_digits = 0

                for row in reader:
                    recognized_plate = row.get('plate_number', '').strip()
                    if recognized_plate:
                        recognized_digits = extract_digits(recognized_plate)
                        common_digits = count_common_consecutive_digits(target_digits, recognized_digits)

                        if common_digits >= 5 and common_digits > max_common_digits:
                            max_common_digits = common_digits
                            best_match = {
                                'image_path': os.path.join(OUTPUT_IMAGES_DIR, train_name, row['image_name']),
                                'plate_number': full_carriage_number,
                                'timestamp': row['timestamp'],
                            }

                if best_match:
                    return best_match

            return {
                'image_path': None,
                'plate_number': full_carriage_number,
                'timestamp': "未知",
            }
    except Exception as e:
        print(f"查找侧面图片信息时出错: {str(e)}")
        return {
            'image_path': None,
            'plate_number': full_carriage_number,
            'timestamp': "未知",
        }


def extract_digits(text):
    """提取文本中的数字序列"""

    import re

    return ''.join(re.findall(r'\d', text))


def count_common_consecutive_digits(digits1, digits2):
    """计算两个数字序列中最长的连续相同数字个数"""

    if not digits1 or not digits2:
        return 0

    max_length = 0
    for i in range(len(digits1)):
        for j in range(len(digits2)):
            length = 0
            while (
                i + length < len(digits1)
                and j + length < len(digits2)
                and digits1[i + length] == digits2[j + length]
            ):
                length += 1
            max_length = max(max_length, length)

    return max_length


def read_train_details_and_insert_data():
    """读取列车详细信息（车厢数据）并插入数据库"""

    train_dir = "F:\\baowen"  # 与 guanggang_main.py 中保持一致
    output_dir = RUNS_OUTPUT_IMAGES_DIR

    if not os.path.exists(train_dir):
        print(f"列车信息目录不存在: {train_dir}")
        return False

    txt_files = [
        f
        for f in os.listdir(train_dir)
        if f.lower().endswith('.txt') and os.path.isfile(os.path.join(train_dir, f))
    ]

    if not txt_files:
        print("未找到列车信息文件")
        return False

    processed_count = 0

    for txt_file in txt_files:
        try:
            train_name = os.path.splitext(txt_file)[0]
            file_path = os.path.join(train_dir, txt_file)

            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)

                normal_carriages = []
                special_carriages = []

                for row in reader:
                    if len(row) >= 5:
                        carriage_id = row[0]
                        carriage_type = row[1].strip()
                        carriage_number = row[2].strip()
                        carriage_time = row[3].strip()

                        full_carriage_number = f"{carriage_type}{carriage_number}"

                        is_special = full_carriage_number.upper().startswith(('P', 'N'))

                        top_image_info = find_image_for_carriage_with_csv_cached(
                            train_name, carriage_id, output_dir
                        )

                        side_image_info = find_side_image_by_plate_number_cached(
                            train_name, full_carriage_number
                        )

                        recognized_plate = "未识别"
                        recognized_time = "未知"

                        if side_image_info:
                            recognized_plate = side_image_info['plate_number']
                            recognized_time = side_image_info['timestamp']

                        carriage_info = {
                            'carriage_id': carriage_id,
                            'full_carriage_number': full_carriage_number,
                            'recognized_plate': recognized_plate,
                            'recognized_time': recognized_time,
                            'top_image_info': top_image_info,
                            'side_image_info': side_image_info,
                            'is_special': is_special,
                        }

                        if is_special:
                            special_carriages.append(carriage_info)
                        else:
                            normal_carriages.append(carriage_info)

                all_carriages = normal_carriages + special_carriages

                for carriage_info in all_carriages:
                    top_image_info = carriage_info['top_image_info']
                    side_image_info = carriage_info['side_image_info']

                    water_info = '有水' if top_image_info.get('has_water', False) else '无水'
                    water_area_ratio = top_image_info.get('water_area_ratio', 0.00)

                    carriage_data = {
                        'carriage_id': carriage_info['carriage_id'],
                        'full_carriage_number': carriage_info['full_carriage_number'],
                        'recognized_plate': carriage_info['recognized_plate'],
                        'recognition_time': carriage_info['recognized_time'],
                        'is_special': carriage_info['is_special'],
                        'water_info': water_info,
                        'water_area_ratio': water_area_ratio,
                        'top_image_path': top_image_info.get('water_detected_path')
                        or top_image_info.get('original_path', ''),
                        'side_image_path': (
                            side_image_info['image_path']
                            if side_image_info and side_image_info.get('image_path')
                            else ''
                        ),
                    }

                    insert_carriage_data(train_name, carriage_data)

            processed_count += 1
            print(f"已处理列车: {train_name}")

        except Exception as e:
            print(f"处理列车文件 {txt_file} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"列车详细信息导入完成，共处理 {processed_count} 趟列车")
    return True


def read_rate_csv_and_insert_data():
    """
    读取rate.csv文件并插入数据库
    """
    db_config = load_database_config()
    if not db_config:
        print("无法加载数据库配置")
        return False
        
    connection = None
    try:
        # 连接数据库
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
        
        # CSV文件路径
        csv_path = os.path.join(DATA_DIR, 'rate.csv')
        
        if not os.path.exists(csv_path):
            print(f"CSV文件不存在: {csv_path}")
            return False
            
        # 读取CSV文件
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            with connection.cursor() as cursor:
                for row in reader:
                    try:
                        # 提取数据
                        train_name = row.get('列车名称', '').strip()
                        total_carriages = int(row.get('总车厢数', 0))
                        correct_count = int(row.get('识别正确数', 0))
                        participated_count = int(row.get('参与统计数', 0))
                        
                        # 处理准确率字段，可能包含百分号
                        accuracy_rate_str = row.get('准确率(%)', '0.00').strip().rstrip('%')
                        accuracy_rate = float(accuracy_rate_str) if accuracy_rate_str else 0.0
                        
                        # 插入或更新列车数据
                        if train_name:
                            insert_train_data(train_name, total_carriages, accuracy_rate, correct_count, participated_count)
                            
                    except ValueError as e:
                        print(f"处理CSV行数据时出错: {e}")
                        continue
                    except Exception as e:
                        print(f"插入列车数据时出错: {e}")
                        continue
        
        connection.commit()
        print("rate.csv数据导入完成")
        return True
                
    except Exception as e:
        print(f"读取rate.csv并插入数据失败: {e}")
        return False
    finally:
        try:
            if connection:
                connection.close()
                print("数据库连接已关闭")
        except:
            pass

def monitor_csv_and_update_database():
    """
    监控CSV文件变化并自动更新数据库
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'rate.csv')
    last_modified = 0
    
    print("开始监控CSV文件变化...")
    
    while True:
        try:
            if os.path.exists(csv_path):
                current_modified = os.path.getmtime(csv_path)
                
                if current_modified > last_modified:
                    print(f"检测到CSV文件更新: {datetime.now()}")
                    read_rate_csv_and_insert_data()
                    # 同时更新列车详细信息
                    read_train_details_and_insert_data()
                    last_modified = current_modified
                    
            # 每30秒检查一次
            time.sleep(30)
            
        except Exception as e:
            print(f"监控过程中出错: {e}")
            time.sleep(30)

def query_recent_water_carriages(days=7, limit=100):
    """
    查询最近有积水的车厢
    
    Args:
        days (int): 查询最近N天的数据（0表示全部）
        limit (int): 返回记录数限制
    
    Returns:
        list: 积水车厢记录列表，每条记录包含：
            - train_name: 列车名
            - carriage_id: 车厢号
            - full_carriage_number: 完整车号
            - water_area_ratio: 积水面积占比
            - top_image_path: 顶部积水图片路径
            - side_image_path: 侧面图片路径
            - recognition_time: 检测时间
    """
    db_config = load_database_config()
    if not db_config:
        print("无法加载数据库配置")
        return []
    
    connection = None
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            if days > 0:
                # 查询指定天数内的数据
                # recognition_time 是 VARCHAR 类型，需要转换为日期时间进行比较
                sql = """
                SELECT 
                    t.train_name,
                    c.carriage_id,
                    c.full_carriage_number,
                    c.water_area_ratio,
                    c.top_image_path,
                    c.side_image_path,
                    c.recognition_time
                FROM carriages c
                JOIN trains t ON c.train_id = t.id
                WHERE c.water_info = '有水'
                  AND c.recognition_time != ''
                  AND c.recognition_time != '未知'
                  AND STR_TO_DATE(c.recognition_time, '%%Y-%%m-%%d %%H:%%i:%%s') >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY STR_TO_DATE(c.recognition_time, '%%Y-%%m-%%d %%H:%%i:%%s') DESC
                LIMIT %s
                """
                cursor.execute(sql, (days, limit))
            else:
                # 查询全部数据
                # recognition_time 是 VARCHAR 类型，需要转换为日期时间进行排序
                sql = """
                SELECT 
                    t.train_name,
                    c.carriage_id,
                    c.full_carriage_number,
                    c.water_area_ratio,
                    c.top_image_path,
                    c.side_image_path,
                    c.recognition_time
                FROM carriages c
                JOIN trains t ON c.train_id = t.id
                WHERE c.water_info = '有水'
                  AND c.recognition_time != ''
                  AND c.recognition_time != '未知'
                ORDER BY STR_TO_DATE(c.recognition_time, '%%Y-%%m-%%d %%H:%%i:%%s') DESC
                LIMIT %s
                """
                cursor.execute(sql, (limit,))
            
            results = cursor.fetchall()
            return list(results)
            
    except pymysql.MySQLError as e:
        print(f"查询积水车厢数据失败: {e}")
        return []
    except Exception as e:
        print(f"查询积水车厢时出错: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        try:
            if connection:
                connection.close()
        except:
            pass

def review_update_carriages(items, default_action=None):
    """
    批量人工审核更新车厢记录（不新增表）。

    参数:
        items: 列表，每项为字典，例如：
            {"train_name": "<列车名>", "carriage_id": "<序号>", "action": "no_water|water"}
            如未提供每项 action，可通过 default_action 指定统一操作。
        default_action: 可选，'no_water' 或 'water'，对未显式指定 action 的项生效。

    规则:
        - 标记为无水(no_water):
            water_info = '无水'
            water_area_ratio = 0.0
            top_image_path = original_path (若存在，否则置为空字符串)
        - 确认为有水(water):
            water_info = '有水'
            water_area_ratio = CSV 占比×1.3 的值（find_image_for_carriage_with_csv_cached 已处理），若取不到则保持0.0
            top_image_path = water_detected_path（若不存在则回退 original_path，仍不存在则置空）

    返回:
        dict: {
            "updated": [ {train_name, carriage_id}... ],
            "failed":  [ {train_name, carriage_id, reason}... ]
        }
    """
    db_config = load_database_config()
    if not db_config:
        return {"updated": [], "failed": [{"reason": "无法加载数据库配置"}]}

    connection = None
    updated, failed = [], []

    try:
        connection = pymysql.connect(
            host=db_config['host'],
            port=int(db_config['port']),
            user=db_config['username'],
            password=str(db_config['password']),
            database=db_config['database_name'],
            charset=db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False
        )

        with connection.cursor() as cursor:
            for item in items or []:
                try:
                    train_name = str(item.get('train_name', '')).strip()
                    carriage_id = str(item.get('carriage_id', '')).strip()
                    full_carriage_number = str(item.get('full_carriage_number', '') or '').strip()
                    action = (item.get('action') or default_action or '').strip().lower()

                    if not train_name or not carriage_id or action not in ('no_water', 'water'):
                        failed.append({"train_name": train_name, "carriage_id": carriage_id, "reason": "参数不完整或action非法"})
                        continue

                    # 解析图片与占比
                    info = find_image_for_carriage_with_csv_cached(train_name, carriage_id, output_dir="runs/output_images")
                    original_path = info.get('original_path') if info else None
                    water_detected_path = info.get('water_detected_path') if info else None
                    try:
                        water_area_ratio = float(info.get('water_area_ratio') or 0.0) if info else 0.0
                    except Exception:
                        water_area_ratio = 0.0

                    if action == 'no_water':
                        set_water_info = '无水'
                        set_ratio = 0.0
                        set_top_path = original_path if original_path and os.path.exists(original_path) else ''
                    else:  # 'water'
                        set_water_info = '有水'
                        set_ratio = water_area_ratio
                        candidate_path = water_detected_path if water_detected_path and os.path.exists(water_detected_path) else original_path
                        set_top_path = candidate_path if candidate_path and os.path.exists(candidate_path) else ''

                    sql = (
                        "UPDATE carriages c JOIN trains t ON c.train_id = t.id "
                        "SET c.water_info=%s, c.water_area_ratio=%s, c.top_image_path=%s, c.updated_at=CURRENT_TIMESTAMP "
                        "WHERE t.train_name=%s AND c.carriage_id=%s"
                    )
                    params = (set_water_info, set_ratio, set_top_path, train_name, carriage_id)
                    cursor.execute(sql, params)

                    if cursor.rowcount <= 0:
                        if full_carriage_number:
                            sql_full = (
                                "UPDATE carriages c JOIN trains t ON c.train_id = t.id "
                                "SET c.water_info=%s, c.water_area_ratio=%s, c.top_image_path=%s, c.updated_at=CURRENT_TIMESTAMP "
                                "WHERE t.train_name=%s AND c.full_carriage_number=%s"
                            )
                            params_full = (set_water_info, set_ratio, set_top_path, train_name, full_carriage_number)
                            cursor.execute(sql_full, params_full)

                        if cursor.rowcount <= 0:
                            failed.append({
                                "train_name": train_name,
                                "carriage_id": carriage_id,
                                "reason": "未匹配到记录(按序号/完整车号均失败)"
                            })
                            continue

                    updated.append({"train_name": train_name, "carriage_id": carriage_id})
                except Exception as e:
                    failed.append({"train_name": item.get('train_name'), "carriage_id": item.get('carriage_id'), "reason": str(e)})

        connection.commit()
        return {"updated": updated, "failed": failed}

    except Exception as e:
        try:
            if connection:
                connection.rollback()
        except:
            pass
        failed.append({"reason": f"数据库或事务错误: {str(e)}"})
        return {"updated": updated, "failed": failed}
    finally:
        try:
            if connection:
                connection.close()
        except:
            pass

def review_mark_no_water(items):
    """批量标记为无水：items 为 [{train_name, carriage_id}, ...]"""
    enriched = [{
        "train_name": it.get('train_name'),
        "carriage_id": it.get('carriage_id'),
        "full_carriage_number": it.get('full_carriage_number'),
        "action": 'no_water'
    } for it in (items or [])]
    return review_update_carriages(enriched)


def review_confirm_water(items):
    """批量确认有水：items 为 [{train_name, carriage_id}, ...]"""
    enriched = [{
        "train_name": it.get('train_name'),
        "carriage_id": it.get('carriage_id'),
        "full_carriage_number": it.get('full_carriage_number'),
        "action": 'water'
    } for it in (items or [])]
    return review_update_carriages(enriched)

def start_auto_database_update():
    """
    启动自动数据库更新服务
    """
    # 首先创建表结构
    if not create_database_tables():
        print("创建数据库表失败")
        return False
        
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_csv_and_update_database, daemon=True)
    monitor_thread.start()
    print("自动数据库更新服务已启动")
    return True

if __name__ == "__main__":
    print("开始测试MySQL数据库连接...")
    success = test_mysql_connection()
    if success:
        print("测试完成!")
        
        # 创建表结构
        print("\n创建数据库表结构...")
        if create_database_tables():
            print("数据库表结构创建成功!")
            
            # 导入现有数据
            print("\n导入现有CSV数据...")
            if read_rate_csv_and_insert_data():
                print("CSV数据导入成功!")
                
                # 导入列车详细信息
                print("\n导入列车详细信息...")
                if read_train_details_and_insert_data():
                    print("列车详细信息导入成功!")
                    
                    # 启动自动更新服务
                    print("\n启动自动数据库更新服务...")
                    if start_auto_database_update():
                        print("自动数据库更新服务运行中... 按Ctrl+C停止")
                        try:
                            # 保持主线程运行
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print("\n服务已停止")
                else:
                    print("列车详细信息导入失败!")
            else:
                print("CSV数据导入失败!")
        else:
            print("数据库表结构创建失败!")
    else:
        print("测试失败!")
        sys.exit(1)