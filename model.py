import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class MedicineRecommender:
    def __init__(self, dataset_path='dataset.csv'):
        self.dataset_path = dataset_path
        self.dataset = None
        # Load the SentenceTransformer model (Layer 3)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_dataset()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()

    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['symptoms'] = self.dataset['symptoms'].fillna('')
        self.dataset['category'] = self.dataset['category'].fillna('general')
        
        # Pre-calculate embeddings for dataset symptoms
        cleaned_symptoms = self.dataset['symptoms'].apply(self.clean_text).tolist()
        self.dataset['embeddings'] = list(self.model.encode(cleaned_symptoms, convert_to_tensor=True))

    def detect_category(self, text):
        """
        LAYER 2: CATEGORY DETECTION
        Filter dataset by detecting categories via keywords.
        """
        # Emergency conditions
        if any(word in text for word in ['heart attack', 'stroke', 'meningitis', 'appendicitis', 'emergency']):
            return 'general'
        
        # Musculoskeletal - very specific
        if any(word in text for word in ['back', 'neck', 'joint', 'muscle', 'cramp', 'arthritis', 'spine', 'spinal', 'stiff', 'shoulder']):
            return 'musculoskeletal'
            
        # Dental - very specific
        if any(word in text for word in ['tooth', 'gum', 'dental', 'jaw', 'mouth', 'canker', 'ulcer', 'toothache']):
            return 'dental'
            
        # Urinary - very specific
        if any(word in text for word in ['urine', 'urination', 'burning urine', 'kidney', 'uti', 'bladder']):
            return 'urinary'
            
        # Respiratory - very specific
        if any(word in text for word in ['cough', 'cold', 'breath', 'wheezing', 'nose', 'sniffle', 'asthma', 'throat', 'sore throat', 'phlegm', 'congestion']):
            return 'respiratory'
            
        # GI - very specific
        if any(word in text for word in ['stomach', 'acidity', 'diarrhea', 'nausea', 'vomit', 'belly', 'digestion', 'gas', 'bloating', 'heartburn', 'constipation', 'bowel']):
            return 'gastrointestinal'
            
        # Neurological - very specific
        if any(word in text for word in ['headache', 'migraine', 'nerve', 'insomnia', 'vertigo', 'dizzy', 'dizziness', 'sleep', 'brain', 'stroke']):
            return 'neurological'
            
        # General skin conditions
        if any(word in text for word in ['skin', 'rash', 'allergy', 'itchy', 'hive', 'burn', 'wound', 'fungal', 'dandruff']):
            return 'general'
            
        return None

    def _semantic_match(self, text, subset_df):
        """
        LAYER 3: SEMANTIC MATCHING
        Computes cosine similarity against the provided dataset subset.
        """
        user_embedding = self.model.encode([text], convert_to_tensor=True)
        
        best_score = -1
        best_idx = -1
        
        all_scores = []
        indices = []
        
        for idx, row in subset_df.iterrows():
            score = util.cos_sim(user_embedding, row['embeddings']).item()
            all_scores.append(score)
            indices.append(idx)
            if score > best_score:
                best_score = score
                best_idx = idx
                
        if all_scores:
            # --- KNN-STYLE RETRIEVAL START ---
            similarity = np.array(all_scores)
            
            # Extract top K matches (K = 3)
            # Sort similarity scores and get indices of top 3 highest values
            k = min(3, len(similarity))
            top_k_indices = np.argsort(similarity)[-k:][::-1]
            
            top_k_matches = []
            categories_found = set()
            
            print("\n--- KNN Top K Matches ---")
            for i in top_k_indices:
                real_idx = indices[i]
                row = subset_df.loc[real_idx]
                match_data = {
                    'symptoms': row['symptoms'],
                    'medicine': row['medicine'],
                    'score': similarity[i],
                    'category': row['category']
                }
                top_k_matches.append(match_data)
                categories_found.add(row['category'])
                
                print(f"- Symptoms: '{match_data['symptoms']}' | Medicine: '{match_data['medicine']}' | Score: {match_data['score']:.4f}")
            
            # Optional Validation (Safe)
            if len(categories_found) > 1:
                print("WARNING: Top K matches belong to inconsistent categories!")
            else:
                print("Validation: Top K matches belong to consistent category.")
            print("-------------------------\n")
            # --- KNN-STYLE RETRIEVAL END ---
                
        if best_idx != -1:
            return subset_df.loc[best_idx], best_score
        return None, 0

    def recommend(self, user_symptoms):
        if not user_symptoms.strip():
            return self._low_confidence_response()

        cleaned_user_symptoms = self.clean_text(user_symptoms)

        # LAYER 1: RULE-BASED PRIORITY (MOST IMPORTANT)
        # Ordered by specificity - most specific first
        # This ensures more detailed multi-word matches are tried before single-word ones
        
        # SPECIFIC MULTI-WORD RULES (HIGHEST PRIORITY)
        specific_multi_word_rules = {
            # Emergency conditions (highest priority)
            "heart attack": ("Emergency Hospital Visit", "CRITICAL: Possible heart attack symptoms.", "Do NOT drive yourself. Call ambulance chew aspirin if doctor advised.", 100.0),
            "chest pain arm numbness": ("Emergency Hospital Visit", "CRITICAL: Possible heart attack symptoms.", "Do NOT drive yourself. Call ambulance chew aspirin if doctor advised.", 100.0),
            "stroke": ("Emergency Hospital Visit", "CRITICAL: Possible stroke symptoms.", "Time is brain. Rush to ER immediately.", 100.0),
            "meningitis": ("Emergency Hospital Visit", "CRITICAL: Possible meningitis symptoms.", "Requires immediate IV antibiotics and medical evaluation.", 100.0),
            "appendicitis": ("Emergency Hospital Visit", "CRITICAL: Possible appendicitis.", "Do not eat or drink anything go to ER to rule out rupture.", 100.0),
            
            # More specific distinguishing rules (to differentiate similar medicines) 
            "high fever body ache chills": ("Paracetamol", "Common analgesic and antipyretic to reduce fever and relieve minor pains.", "Take after food. Avoid alcohol. Max 4 doses in 24 hours.", 100.0),
            "mild fever extreme fatigue sore muscles": ("Ibuprofen", "NSAID used for reducing high fever and relieving body inflammation.", "Take with a meal. Do not use if you have stomach ulcers.", 100.0),
            "fever sore throat cough": ("Paracetamol / Vitamin C", "Supportive treatment for viral fever and cold symptoms.", "Rest adequately keep hydrated isolate if contagious.", 100.0),
            "weight loss thirst diabetes": ("Consult Endocrinologist", "Symptoms pointing to suspected diabetes.", "Fast for 8 hours and get a blood sugar test done.", 100.0),
            "cold sore blister": ("Acyclovir Cream", "Antiviral for herpes simplex cold sores.", "Apply at earliest sign of tingling. Wash hands after use.", 100.0),
            "car sick motion": ("Dimenhydrinate", "Prevents nausea associated with vehicle motion.", "Take 1 hour before travel. Will cause drowsiness.", 100.0),
            "seasonal allergy pollen": ("Fexofenadine", "Non-drowsy antihistamine for allergy symptoms.", "Avoid fruit juices like grapefruit or apple while taking.", 100.0),
            "pale skin chronic fatigue anemia iron": ("Iron Supplement", "Treats iron deficiency anemia.", "Take with orange juice. May cause dark stools.", 100.0),
            "iron deficiency": ("Iron Supplement", "Treats iron deficiency anemia.", "Take with orange juice. May cause dark stools.", 100.0),
            "upper back neck shoulder": ("Diclofenac Gel / Ibuprofen", "Relieves upper back and neck muscular pain.", "Avoid awkward sleeping postures. Use a supportive pillow.", 100.0),
        }
        
        # GENERAL SINGLE-WORD and COMMON MULTI-WORD RULES
        general_rules = {
            # Dental conditions
            "toothache": ("Clove Oil / Benzocaine Gel", "Local anesthetic and counter-irritant for dental pain.", "Apply locally on tooth cavity do not swallow. Visit dentist.", 100.0),
            "tooth pain": ("Clove Oil / Benzocaine Gel", "Local anesthetic and counter-irritant for dental pain.", "Apply locally on tooth cavity do not swallow. Visit dentist.", 100.0),
            "sensitive teeth": ("Desensitizing Toothpaste", "Reduces tooth sensitivity by blocking exposed dentin channels.", "Brush gently twice a day. Avoid acidic juices.", 100.0),
            "mouth ulcer": ("Choline Salicylate Gel", "Numbs mouth ulcers for temporary relief.", "Apply small amount on ulcer before meals.", 100.0),
            "bad breath": ("Chlorhexidine Mouthwash", "Antibacterial rinse to improve oral hygiene.", "Use after brushing do not swallow maintain dental care.", 100.0),
            
            # Respiratory conditions
            "dry cough": ("Dextromethorphan Syrup", "Cough suppressant for non-productive dry cough.", "May cause drowsiness avoid driving. Follow dosage cup.", 100.0),
            "wet cough": ("Guaifenesin Syrup", "Expectorant to loosen chest congestion and mucus.", "Drink plenty of water to help thin mucus.", 100.0),
            "cough phlegm": ("Guaifenesin Syrup", "Expectorant to loosen chest congestion and mucus.", "Drink plenty of water to help thin mucus.", 100.0),
            "sore throat": ("Lozenges / Paracetamol", "Relief for inflamed throat tissues and pain.", "Avoid cold drinks talk less consult doctor if white patches appear.", 100.0),
            "runny nose": ("Cetirizine", "Antihistamine for seasonal allergies and running nose.", "May cause mild drowsiness. Take at night.", 100.0),
            "nasal congestion": ("Cetirizine", "Antihistamine for seasonal allergies and running nose.", "May cause mild drowsiness. Take at night.", 100.0),
            "blocked nose": ("Pseudoephedrine", "Decongestant for clearing blocked nasal passages.", "Can increase heart rate use with caution if hypertensive.", 100.0),
            "stuffy nose": ("Pseudoephedrine", "Decongestant for clearing blocked nasal passages.", "Can increase heart rate use with caution if hypertensive.", 100.0),
            "asthma": ("Salbutamol Inhaler", "Bronchodilator to open airways quickly.", "Use only as prescribed. Keep inhaler clean.", 100.0),
            "wheezing": ("Salbutamol Inhaler", "Bronchodilator to open airways quickly.", "Use only as prescribed. Keep inhaler clean.", 100.0),
            "shortness of breath": ("Salbutamol Inhaler", "Bronchodilator to open airways quickly.", "Use only as prescribed. Keep inhaler clean.", 100.0),
            
            # GI conditions
            "acid reflux": ("Antacid Gel (Magaldrate/Simethicone)", "Neutralizes stomach acid and relieves gas.", "Take 30 mins after meals. Chew properly if tablet.", 100.0),
            "heartburn": ("Antacid Gel (Magaldrate/Simethicone)", "Neutralizes stomach acid and relieves gas.", "Take 30 mins after meals. Chew properly if tablet.", 100.0),
            "stomach ache": ("Domperidone / Pantoprazole", "Prokinetic and acid reducer for upper GI discomfort.", "Take 30 minutes before breakfast.", 100.0),
            "diarrhea": ("Loperamide", "Anti-diarrheal medication to slow bowel movements.", "Stop once stools harden. Drink oral rehydration salts (ORS).", 100.0),
            "nausea vomiting": ("Ondansetron", "Anti-emetic to prevent or stop nausea and vomiting.", "Place tablet under tongue or swallow with water.", 100.0),
            "gas bloating": ("Simethicone", "Anti-foaming agent to break up gas bubbles in the gut.", "Chew tablets thoroughly. Avoid gas-producing foods.", 100.0),
            "constipation": ("Bisacodyl / Psyllium Husk", "Laxative or roughage to stimulate bowel movement.", "Drink plenty of water. Increase fiber in daily diet.", 100.0),
            "hemorrhoids": ("Hydrocortisone Suppository", "Reduces inflammation and pain in anal area.", "Eat high fiber diet avoid excessive straining on toilet.", 100.0),
            
            # Urinary conditions
            "burning urination": ("Alkalizer (Disodium Hydrogen Citrate)", "Urine alkalizer for soothing urinary tract irritation.", "Mix syrup in a full glass of water. Consult doctor for antibiotics.", 100.0),
            "kidney stone": ("Analgesic / Antispasmodic", "Pain relief for suspected kidney stone pain.", "Drink excessive water immediately seek medical care.", 100.0),
            "uti": ("Cranberry Extract / Alkalizer", "Natural relief and pH balancer for urinary discomfort.", "Increase water intake substantially. Seek culture text.", 100.0),
            
            # Neurological conditions
            "insomnia": ("Melatonin", "Supplement to regulate sleep-wake cycles.", "Take 30 mins before desired sleep time suppress room light.", 100.0),
            "migraine": ("Sumatriptan", "Specific medication for migraine attacks.", "Take at onset of migraine pain.", 100.0),
            "nerve pain": ("Gabapentin / Carbamazepine", "Prescription relief for severe nerve-related pain.", "Drowsiness expected. Prescription only consult doctor.", 100.0),
            "dizziness": ("Meclizine / Betahistine", "Treats instances of vertigo and balance disorders.", "Do not operate heavy machinery. Move slowly from seated positions.", 100.0),
            "vertigo": ("Meclizine / Betahistine", "Treats instances of vertigo and balance disorders.", "Do not operate heavy machinery. Move slowly from seated positions.", 100.0),
            
            # Musculoskeletal conditions
            "back pain": ("Muscle Relaxant / Ibuprofen", "Relieves musculoskeletal pain and spasm in the back.", "Apply hot compress rest on firm mattress. Maintain good posture.", 100.0),
            "neck pain": ("Diclofenac Gel / Ibuprofen", "Relieves upper back and neck muscular pain.", "Avoid awkward sleeping postures. Use a supportive pillow.", 100.0),
            "joint pain": ("Diclofenac Gel / Tablet", "Strong NSAID for joint inflammation and arthritis.", "Apply topically first. If oral take strictly after meals.", 100.0),
            "arthritis": ("Diclofenac Gel / Tablet", "Strong NSAID for joint inflammation and arthritis.", "Apply topically first. If oral take strictly after meals.", 100.0),
            "muscle cramp": ("Magnesium Supplement", "Helps in muscle relaxation and preventing cramps.", "Stay hydrated stretch muscles gently before sleep.", 100.0),
            "muscle ache": ("Epsom Salt Bath / Ibuprofen", "Reduces muscular tension and inflammation.", "Rest and hydrate. Gently stretch affected muscles.", 100.0),
            
            # Skin conditions
            "itchy skin": ("Hydrocortisone Cream", "Topical steroid for suppressing allergic skin reactions.", "Apply thin layer. Do not use on broken skin.", 100.0),
            "hives": ("Hydrocortisone Cream", "Topical steroid for suppressing allergic skin reactions.", "Apply thin layer. Do not use on broken skin.", 100.0),
            "fungal infection": ("Clotrimazole Cream", "Topical anti-fungal to kill fungi on skin.", "Wash and dry area completely before applying.", 100.0),
            "athlete foot": ("Clotrimazole Cream", "Topical anti-fungal to kill fungi on skin.", "Wash and dry area completely before applying.", 100.0),
            "dandruff": ("Ketoconazole Shampoo", "Medicated shampoo for severe fungal dandruff.", "Leave lather on scalp for 5 mins before rinsing.", 100.0),
            
            # Eye conditions
            "eye discharge": ("Antibiotic Eye Drops", "Treatment for bacterial conjunctivitis (pink eye).", "Do not touch dropper tip to eye. Wash hands frequently.", 100.0),
            "conjunctivitis": ("Antibiotic Eye Drops", "Treatment for bacterial conjunctivitis (pink eye).", "Do not touch dropper tip to eye. Wash hands frequently.", 100.0),
            "pink eye": ("Antibiotic Eye Drops", "Treatment for bacterial conjunctivitis (pink eye).", "Do not touch dropper tip to eye. Wash hands frequently.", 100.0),
            "dry eyes": ("Artificial Tears", "Lubricating eye drops to relieve dry irritated eyes.", "Blink frequently while using screens. Use drops as needed.", 100.0),
            
            # General/misc conditions
            "fever": ("Paracetamol", "Common analgesic and antipyretic to reduce fever and relieve minor pains.", "Take after food. Avoid alcohol. Max 4 doses in 24 hours.", 100.0),
            "headache": ("Aspirin / Paracetamol", "Pain reliever for tension or mild migraine headaches.", "Take with milk or food. Do not give to children under 16.", 100.0),
            "insomnia sleep": ("Melatonin", "Supplement to regulate sleep-wake cycles.", "Take 30 mins before desired sleep time suppress room light.", 100.0),
            "earache": ("Analgesic Ear Drops", "Relieves mild congestive ear pain.", "Do not put drops if eardrum is ruptured.", 100.0),
            "burn": ("Aloe Vera / Silver Sulfadiazine", "Soothes thermal burns and prevents infection.", "Cool under running water first. Do not burst blisters.", 100.0),
            "wound": ("Povidone Iodine Ointment", "Antiseptic to clean wounds and prevent bacterial growth.", "Clean wound with water apply ointment and bandage.", 100.0),
            "cold sore": ("Acyclovir Cream", "Antiviral for herpes simplex cold sores.", "Apply at earliest sign of tingling. Wash hands after use.", 100.0),
            "insect sting": ("Antihistamine / Ice Pack", "Reduces histamine response to insect venom.", "Remove stinger if present. Wash with soap and water.", 100.0),
            "motion sickness": ("Dimenhydrinate", "Prevents nausea associated with vehicle motion.", "Take 1 hour before travel. Will cause drowsiness.", 100.0),
            "anemia": ("Iron Supplement", "Treats iron deficiency anemia.", "Take with orange juice. May cause dark stools.", 100.0),
            "menstrual": ("Mefenamic Acid / Drotaverine", "Relieves smooth muscle spasms and period pain.", "Use a heating pad. Take medication preferably after food.", 100.0),
        }
        
        # Try specific multi-word rules first (highest priority)
        for key, value in specific_multi_word_rules.items():
            key_words = key.split()
            # Try exact multi-word match
            if re.search(r'\b' + re.escape(key) + r'\b', cleaned_user_symptoms):
                return {
                    "medicine": value[0],
                    "description": value[1],
                    "precautions": value[2],
                    "score": value[3],
                    "confidence": value[3]
                }
            # Score based on keyword overlap for multi-word rules
            if len(key_words) > 1:
                word_matches = sum(1 for word in key_words if word in cleaned_user_symptoms.split())
                match_ratio = word_matches / len(key_words)
                # If at least 66% of keywords match for specific rules, use it
                if match_ratio >= 0.66:
                    return {
                        "medicine": value[0],
                        "description": value[1],
                        "precautions": value[2],
                        "score": value[3],
                        "confidence": value[3]
                    }
        
        # Try general rules (normal priority)
        best_rule_match = None
        best_match_score = 0
        
        for key, value in general_rules.items():
            # Try exact multi-word match first
            if re.search(r'\b' + re.escape(key) + r'\b', cleaned_user_symptoms):
                return {
                    "medicine": value[0],
                    "description": value[1],
                    "precautions": value[2],
                    "score": value[3],
                    "confidence": value[3]
                }
            
            # Score based on keyword overlap for multi-word rules
            key_words = key.split()
            if len(key_words) > 1:
                word_matches = sum(1 for word in key_words if word in cleaned_user_symptoms.split())
                match_ratio = word_matches / len(key_words)
                
                # If at least 50% of keywords match, it's a candidate
                if match_ratio >= 0.5 and match_ratio > best_match_score:
                    best_match_score = match_ratio
                    best_rule_match = (value, match_ratio)
            
            # Try partial match for single keywords
            elif len(key_words) == 1 and key in cleaned_user_symptoms:
                return {
                    "medicine": value[0],
                    "description": value[1],
                    "precautions": value[2],
                    "score": value[3],
                    "confidence": value[3]
                }
        
        # Apply best partial match if found
        if best_rule_match and best_match_score >= 0.5:
            value, match_ratio = best_rule_match
            return {
                "medicine": value[0],
                "description": value[1],
                "precautions": value[2],
                "score": value[3] * match_ratio,
                "confidence": value[3] * match_ratio
            }

        # LAYER 2: CATEGORY DETECTION
        detected_category = self.detect_category(cleaned_user_symptoms)
        
        # Filter dataset or use whole if no specific category detected
        if detected_category:
            subset_df = self.dataset[self.dataset['category'] == detected_category]
        else:
            subset_df = self.dataset
            
        # If somehow category filtering yields empty, fallback to full dataset
        if subset_df.empty:
            subset_df = self.dataset

        # LAYER 3: SEMANTIC MATCHING
        best_row, best_score = self._semantic_match(cleaned_user_symptoms, subset_df)

        # Confidence Control (from Prompt: < 0.4)
        if best_score < 0.45:
            return self._low_confidence_response()

        return {
            "medicine": best_row['medicine'],
            "description": best_row['description'],
            "precautions": best_row['precautions'],
            "score": round(best_score * 100, 2),
            "confidence": round(best_score * 100, 2)
        }

    def _low_confidence_response(self):
        return {
            "medicine": "No strong match found",
            "description": "Symptoms do not clearly match a known condition",
            "precautions": "Please consult a doctor",
            "score": 0,
            "confidence": 0
        }

    # ============ ACCURACY COMPONENT ============
    
    def _predict_medicine(self, user_symptoms, apply_confidence_threshold=True):
        """
        Internal prediction method that returns predicted medicine.
        Used for evaluation purposes.
        """
        result = self.recommend(user_symptoms)
        
        if apply_confidence_threshold:
            return result['medicine']
        else:
            # Return prediction even if below confidence threshold
            return result['medicine']

    def evaluate(self):
        """
        Evaluate model using Leave-One-Out Cross-Validation (LOOCV).
        Tests the model by predicting each sample's medicine from dataset.
        """
        print("\n" + "="*60)
        print("MODEL ACCURACY EVALUATION")
        print("="*60)
        
        true_labels = []
        predicted_labels = []
        correct_predictions = 0
        total_predictions = 0
        
        print("\nRunning Leave-One-Out Cross-Validation (LOOCV)...")
        print("-" * 60)
        
        for idx, row in self.dataset.iterrows():
            true_medicine = row['medicine']
            symptoms = row['symptoms']
            
            # Make prediction
            result = self.recommend(symptoms)
            predicted_medicine = result['medicine']
            
            true_labels.append(true_medicine)
            predicted_labels.append(predicted_medicine)
            
            if true_medicine == predicted_medicine:
                correct_predictions += 1
                status = "✓"
            else:
                status = "✗"
            
            total_predictions += 1
            
            # Print progress for first 10 and last 5 samples
            if idx < 10 or idx >= len(self.dataset) - 5:
                print(f"{status} [{idx+1}/{len(self.dataset)}] True: {true_medicine[:30]:30} | Pred: {predicted_medicine[:30]:30}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predicted_labels)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print("="*60 + "\n")
        
        return metrics

    def calculate_metrics(self, true_labels, predicted_labels):
        """
        Calculate accuracy, precision, recall, and F1-score.
        """
        acc = accuracy_score(true_labels, predicted_labels)
        
        # Handle multi-class metrics
        try:
            prec = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            rec = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        except Exception as e:
            print(f"Warning: Could not calculate some metrics: {e}")
            prec = rec = f1 = 0
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

    def cross_validate(self, k=5):
        """
        Perform k-fold cross-validation.
        Splits dataset into k folds and evaluates model on each fold.
        """
        print("\n" + "="*60)
        print(f"K-FOLD CROSS-VALIDATION (K={k})")
        print("="*60)
        
        fold_size = len(self.dataset) // k
        fold_scores = []
        
        for fold_num in range(k):
            print(f"\nFold {fold_num + 1}/{k}...")
            
            # Split data into train and test
            test_start = fold_num * fold_size
            test_end = test_start + fold_size if fold_num < k - 1 else len(self.dataset)
            
            test_set = self.dataset.iloc[test_start:test_end]
            
            true_labels = list(test_set['medicine'])
            predicted_labels = []
            
            for idx, row in test_set.iterrows():
                result = self.recommend(row['symptoms'])
                predicted_labels.append(result['medicine'])
            
            metrics = self.calculate_metrics(true_labels, predicted_labels)
            fold_scores.append(metrics)
            
            print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_scores]),
            'precision': np.mean([m['precision'] for m in fold_scores]),
            'recall': np.mean([m['recall'] for m in fold_scores]),
            'f1': np.mean([m['f1'] for m in fold_scores])
        }
        
        print("\n" + "-"*60)
        print("CROSS-VALIDATION RESULTS (Average)")
        print("-"*60)
        print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall:    {avg_metrics['recall']:.4f}")
        print(f"F1-Score:  {avg_metrics['f1']:.4f}")
        print("="*60 + "\n")
        
        return avg_metrics, fold_scores

    def get_accuracy_report(self):
        """
        Generate a comprehensive accuracy report with confusion matrix details.
        """
        print("\nGenerating comprehensive accuracy report...")
        
        true_labels = []
        predicted_labels = []
        
        for idx, row in self.dataset.iterrows():
            result = self.recommend(row['symptoms'])
            true_labels.append(row['medicine'])
            predicted_labels.append(result['medicine'])
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(true_labels, predicted_labels, zero_division=0))
        
        return {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }

if __name__ == "__main__":
    recommender = MedicineRecommender()
    
    print("Testing ML module:")
    print(recommender.recommend("headache and slight fever"))
    
    # Test accuracy components
    print("\n\nTesting accuracy components:")
    print("1. Running evaluation...")
    metrics = recommender.evaluate()
    
    print("\n2. Running k-fold cross-validation...")
    avg_metrics, fold_scores = recommender.cross_validate(k=5)
    
    print("\n3. Generating classification report...")
    report = recommender.get_accuracy_report()
