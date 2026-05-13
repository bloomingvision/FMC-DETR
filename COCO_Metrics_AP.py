import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class DualLogger:
    """同时输出到控制台和文件的流处理器"""
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='instances_val2017.json', help='Annotation COCO JSON path')
    parser.add_argument('--pred_json', type=str, default='predictions.json', help='Prediction COCO JSON path')
    parser.add_argument('--output_txt', type=str, default='coco_results_test.txt', help='Output result file path')
    return parser.parse_known_args()[0]

def save_coco_results(evaluator, file_path):
    """实时显示并保存评估结果"""
    with open(file_path, 'w') as f:
        # 创建双输出流
        dual_logger = DualLogger(sys.stdout, f)
        
        # 重定向标准输出
        with redirect_stdout(dual_logger):
            print("\n[Real-time Evaluation Process]")
            evaluator.summarize()  # 此时输出会同时显示和保存
            
            # 补充结构化指标
            print("\n[Structured Metrics]")
            metrics = [
                ('AP @0.50:0.95', evaluator.stats[0]),
                ('AP50', evaluator.stats[1]),
                ('AP75', evaluator.stats[2]),
                ('AP_small', evaluator.stats[3]),
                ('AP_medium', evaluator.stats[4]),
                ('AP_large', evaluator.stats[5]),
                ('AR_max1', evaluator.stats[6]),
                ('AR_max10', evaluator.stats[7]),
                ('AR_max100', evaluator.stats[8]),
                ('AR_small', evaluator.stats[9]),
                ('AR_medium', evaluator.stats[10]),
                ('AR_large', evaluator.stats[11]),
            ]
            
            # 格式化输出
            for name, value in metrics:
                line = f"{name.ljust(15)}: {value:.4f}"
                print(line)

if __name__ == '__main__':
    opt = parse_opt()
    
    # 初始化COCO API
    coco_gt = COCO(opt.anno_json)
    coco_dt = coco_gt.loadRes(opt.pred_json)
    
    # 执行评估
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # 实时显示并保存结果
    save_coco_results(coco_eval, opt.output_txt)
    print(f"\n✅ Evaluation results saved to {opt.output_txt}")