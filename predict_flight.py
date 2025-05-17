from predict_delays import predict_delay

def get_user_input():
    """Get flight information from user"""
    print("\n=== Flight Delay Prediction ===")
    print("Please enter the following flight details:")
    
    # Get month (1-12)
    while True:
        try:
            month = int(input("\nMonth (1-12): "))
            if 1 <= month <= 12:
                break
            print("Please enter a number between 1 and 12")
        except ValueError:
            print("Please enter a valid number")
    
    # Get day of week (1-7)
    while True:
        try:
            day = int(input("Day of week (1-7, where 1=Monday): "))
            if 1 <= day <= 7:
                break
            print("Please enter a number between 1 and 7")
        except ValueError:
            print("Please enter a valid number")
    
    # Get airline code
    airline = input("Airline code (e.g., AA, UA, DL): ").upper()
    
    # Get airports
    origin = input("Origin airport code (e.g., JFK, LAX): ").upper()
    destination = input("Destination airport code (e.g., JFK, LAX): ").upper()
    
    # Get scheduled departure time
    while True:
        try:
            dep_time = int(input("Scheduled departure time (HHMM, e.g., 800 for 8:00 AM): "))
            if 0 <= dep_time <= 2359:
                break
            print("Please enter a valid time in HHMM format")
        except ValueError:
            print("Please enter a valid number")
    
    # Get distance
    while True:
        try:
            distance = float(input("Flight distance in miles: "))
            if distance > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    return {
        'MONTH': month,
        'DAY_OF_WEEK': day,
        'AIRLINE': airline,
        'ORIGIN_AIRPORT': origin,
        'DESTINATION_AIRPORT': destination,
        'SCHEDULED_DEPARTURE': dep_time,
        'DISTANCE': distance
    }

def main():
    try:
        # Get user input
        flight_data = get_user_input()
        
        # Make prediction
        prediction, probability = predict_delay(flight_data)
        
        # Display results
        print("\n=== Prediction Results ===")
        print(f"Flight Status: {'Delayed' if prediction == 1 else 'Not Delayed'}")
        print(f"Probability of Delay: {probability:.2%}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please make sure you have run the training script first (predict_delays.py)")

if __name__ == "__main__":
    main() 