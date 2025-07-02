"""
API 效能基準測試
測試藥丸檢測 API 的響應時間和吞吐量
"""
import asyncio
import aiohttp
import time
import statistics
import json
import sys
from pathlib import Path

class APIBenchmark:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            'health_check': [],
            'file_detection': [],
            'url_detection': [],
            'concurrent_requests': []
        }
    
    async def benchmark_health_check(self, iterations=50):
        """基準測試：健康檢查端點"""
        print(f"🩺 測試健康檢查端點 ({iterations} 次)...")
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                start_time = time.perf_counter()
                async with session.get(f"{self.base_url}/health") as response:
                    await response.text()
                    response_time = time.perf_counter() - start_time
                    self.results['health_check'].append(response_time * 1000)  # 轉換為毫秒
                
                if (i + 1) % 10 == 0:
                    print(f"   完成 {i + 1}/{iterations}")
    
    async def benchmark_file_detection(self, iterations=10):
        """基準測試：檔案上傳檢測"""
        print(f"🔍 測試檔案檢測端點 ({iterations} 次)...")
        
        test_image_path = Path("tests/image.jpg")
        if not test_image_path.exists():
            print("   ⚠️ 測試圖片不存在，跳過檔案檢測測試")
            return
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                start_time = time.perf_counter()
                
                with open(test_image_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename='test.jpg', content_type='image/jpeg')
                    
                    async with session.post(f"{self.base_url}/detect", data=data) as response:
                        await response.text()
                        response_time = time.perf_counter() - start_time
                        self.results['file_detection'].append(response_time * 1000)
                
                print(f"   完成 {i + 1}/{iterations} (最後響應: {response_time*1000:.1f}ms)")
    
    async def benchmark_url_detection(self, iterations=5):
        """基準測試：URL 檢測"""
        print(f"🌐 測試 URL 檢測端點 ({iterations} 次)...")
        
        test_url = "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg"
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                start_time = time.perf_counter()
                
                data = aiohttp.FormData()
                data.add_field('image_url', test_url)
                
                async with session.post(f"{self.base_url}/detect", data=data) as response:
                    await response.text()
                    response_time = time.perf_counter() - start_time
                    self.results['url_detection'].append(response_time * 1000)
                
                print(f"   完成 {i + 1}/{iterations} (最後響應: {response_time*1000:.1f}ms)")
    
    async def benchmark_concurrent_requests(self, concurrent_users=5, requests_per_user=3):
        """基準測試：並發請求"""
        print(f"⚡ 測試並發請求 ({concurrent_users} 用戶，每人 {requests_per_user} 請求)...")
        
        async def user_requests(session, user_id):
            user_times = []
            for req_id in range(requests_per_user):
                start_time = time.perf_counter()
                async with session.get(f"{self.base_url}/health") as response:
                    await response.text()
                    response_time = time.perf_counter() - start_time
                    user_times.append(response_time * 1000)
            return user_times
        
        async with aiohttp.ClientSession() as session:
            tasks = [user_requests(session, i) for i in range(concurrent_users)]
            all_results = await asyncio.gather(*tasks)
            
            # 合併所有結果
            for user_results in all_results:
                self.results['concurrent_requests'].extend(user_results)
    
    def calculate_stats(self, times):
        """計算統計數據"""
        if not times:
            return {}
        
        return {
            'min': round(min(times), 2),
            'max': round(max(times), 2),
            'mean': round(statistics.mean(times), 2),
            'median': round(statistics.median(times), 2),
            'p95': round(sorted(times)[int(len(times) * 0.95)], 2),
            'p99': round(sorted(times)[int(len(times) * 0.99)], 2),
            'count': len(times)
        }
    
    def print_results(self):
        """列印效能測試結果"""
        print("\n" + "="*60)
        print("📊 API 效能基準測試結果")
        print("="*60)
        
        for test_name, times in self.results.items():
            if not times:
                continue
                
            stats = self.calculate_stats(times)
            print(f"\n🔹 {test_name.replace('_', ' ').title()}")
            print(f"   請求數: {stats['count']}")
            print(f"   平均值: {stats['mean']}ms")
            print(f"   中位數: {stats['median']}ms")
            print(f"   最小值: {stats['min']}ms")
            print(f"   最大值: {stats['max']}ms")
            print(f"   P95:   {stats['p95']}ms")
            print(f"   P99:   {stats['p99']}ms")
    
    def save_results(self, filename="benchmark_results.json"):
        """儲存結果為 JSON 檔案"""
        benchmark_data = {
            'timestamp': time.time(),
            'base_url': self.base_url,
            'results': {}
        }
        
        for test_name, times in self.results.items():
            benchmark_data['results'][test_name] = {
                'raw_times': times,
                'stats': self.calculate_stats(times)
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 結果已儲存至: {filename}")
    
    async def run_all_benchmarks(self):
        """執行所有基準測試"""
        print("🚀 開始 API 效能基準測試")
        print("-" * 40)
        
        try:
            await self.benchmark_health_check()
            await self.benchmark_concurrent_requests()
            await self.benchmark_file_detection()
            await self.benchmark_url_detection()
            
            self.print_results()
            self.save_results()
            
            # 檢查是否有效能問題
            health_stats = self.calculate_stats(self.results['health_check'])
            if health_stats and health_stats['p95'] > 1000:  # P95 > 1 秒
                print("\n⚠️ 警告：健康檢查 P95 響應時間超過 1 秒")
                return False
            
            detection_stats = self.calculate_stats(self.results['file_detection'])
            if detection_stats and detection_stats['p95'] > 10000:  # P95 > 10 秒
                print("\n⚠️ 警告：檢測端點 P95 響應時間超過 10 秒")
                return False
            
            print("\n✅ 所有效能測試通過")
            return True
            
        except Exception as e:
            print(f"\n❌ 基準測試失敗: {e}")
            return False

async def main():
    """主函數"""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    benchmark = APIBenchmark(base_url)
    success = await benchmark.run_all_benchmarks()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())