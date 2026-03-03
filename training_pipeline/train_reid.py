"""
ReID Model Training Script
Train OSNet for person/vehicle re-identification
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAIN_REID")

def train_person_reid(data_dir='training_data/reid_persons', output_dir='weights/reid_person'):
    """
    Train person ReID model using torchreid
    
    Args:
        data_dir: Path to ReID dataset
        output_dir: Path to save trained model
    """
    try:
        import torch
        import torchreid
        
        logger.info("Starting Person ReID training...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data manager
        datamanager = torchreid.data.ImageDataManager(
            root=data_dir,
            sources='custom',  # Custom dataset
            targets='custom',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop'],
            num_instances=4,  # Number of instances per identity in a batch
            train_sampler='RandomIdentitySampler'
        )
        
        # Model
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            loss='softmax',
            pretrained=True
        )
        
        # Optimizer
        optimizer = torchreid.optim.build_optimizer(
            model,
            optim='adam',
            lr=0.0003
        )
        
        # Scheduler
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        
        # Engine
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smooth=True
        )
        
        # Training
        engine.run(
            save_dir=output_dir,
            max_epoch=60,
            eval_freq=10,
            print_freq=10,
            test_only=False
        )
        
        logger.info(f"✓ Training complete. Model saved to: {output_dir}")
        
    except ImportError:
        logger.error("torchreid not installed. Install with: pip install torchreid")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def train_vehicle_reid(data_dir='training_data/reid_vehicles', output_dir='weights/reid_vehicle'):
    """
    Train vehicle ReID model
    Similar to person ReID but optimized for vehicles
    """
    try:
        import torch
        import torchreid
        
        logger.info("Starting Vehicle ReID training...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data manager (same as person but different dataset)
        datamanager = torchreid.data.ImageDataManager(
            root=data_dir,
            sources='custom',
            targets='custom',
            height=256,
            width=256,  # Vehicles are more square than persons
            batch_size_train=32,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop', 'color_jitter'],
            num_instances=4
        )
        
        # Model
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            loss='softmax',
            pretrained=True
        )
        
        # Optimizer
        optimizer = torchreid.optim.build_optimizer(
            model,
            optim='adam',
            lr=0.0003
        )
        
        # Scheduler
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        
        # Engine
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Training
        engine.run(
            save_dir=output_dir,
            max_epoch=60,
            eval_freq=10,
            print_freq=10,
            test_only=False
        )
        
        logger.info(f"✓ Training complete. Model saved to: {output_dir}")
        
    except ImportError:
        logger.error("torchreid not installed. Install with: pip install torchreid")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def evaluate_model(model_path, data_dir, model_type='person'):
    """
    Evaluate trained ReID model
    
    Args:
        model_path: Path to trained model
        data_dir: Path to test dataset
        model_type: 'person' or 'vehicle'
    """
    try:
        import torch
        import torchreid
        
        logger.info(f"Evaluating {model_type} ReID model...")
        
        # Data manager
        height = 256
        width = 128 if model_type == 'person' else 256
        
        datamanager = torchreid.data.ImageDataManager(
            root=data_dir,
            sources='custom',
            targets='custom',
            height=height,
            width=width,
            batch_size_test=100
        )
        
        # Load model
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            loss='softmax',
            pretrained=False
        )
        
        torchreid.utils.load_pretrained_weights(model, model_path)
        
        # Engine
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager,
            model
        )
        
        # Test
        engine.run(
            save_dir='logs/reid_eval',
            max_epoch=0,
            test_only=True
        )
        
        logger.info("✓ Evaluation complete")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ReID models for HAWKEYE')
    parser.add_argument('--type', type=str, choices=['person', 'vehicle', 'both'], 
                       default='person', help='Type of ReID model to train')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='Base directory for training data')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Directory to save trained models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model instead of training')
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        if not args.model_path:
            logger.error("--model-path required for evaluation")
            sys.exit(1)
        
        data_dir = f"{args.data_dir}/reid_persons" if args.type == 'person' else f"{args.data_dir}/reid_vehicles"
        evaluate_model(args.model_path, data_dir, args.type)
    else:
        if args.type in ['person', 'both']:
            train_person_reid(
                data_dir=f"{args.data_dir}/reid_persons",
                output_dir=f"{args.output_dir}/reid_person"
            )
        
        if args.type in ['vehicle', 'both']:
            train_vehicle_reid(
                data_dir=f"{args.data_dir}/reid_vehicles",
                output_dir=f"{args.output_dir}/reid_vehicle"
            )
