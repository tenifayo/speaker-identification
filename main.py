"""CLI interface for speaker identification system."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Speaker Identification System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll a new user
  python main.py enroll --user john --name "John Doe" --audio sample1.wav sample2.wav sample3.wav

  # Verify a speaker
  python main.py verify --user john --audio test.wav

  # Identify unknown speaker
  python main.py identify --audio unknown.wav

  # List enrolled users
  python main.py list

  # Start API server
  python main.py serve --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("--user", "-u", required=True, help="User ID")
    enroll_parser.add_argument("--name", "-n", required=True, help="Display name")
    enroll_parser.add_argument("--audio", "-a", nargs="+", required=True, help="Audio files")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify speaker identity")
    verify_parser.add_argument("--user", "-u", required=True, help="Claimed user ID")
    verify_parser.add_argument("--audio", "-a", required=True, help="Test audio file")
    verify_parser.add_argument("--threshold", "-t", type=float, help="Similarity threshold")
    
    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Identify unknown speaker")
    identify_parser.add_argument("--audio", "-a", required=True, help="Test audio file")
    identify_parser.add_argument("--top", "-n", type=int, default=3, help="Top N matches")
    identify_parser.add_argument("--threshold", "-t", type=float, help="Minimum threshold")
    
    # List command
    subparsers.add_parser("list", help="List enrolled users")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete user enrollment")
    delete_parser.add_argument("--user", "-u", required=True, help="User ID to delete")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Import here to avoid slow startup for --help
    if args.command == "enroll":
        from src.enrollment import enroll_user, EnrollmentError
        try:
            result = enroll_user(args.user, args.name, args.audio)
            print(f"✓ Enrolled user '{result['user_id']}' with {result['num_samples']} samples")
            print(f"  Embedding dimension: {result['embedding_dim']}")
        except EnrollmentError as e:
            print(f"✗ Enrollment failed: {e}")
            sys.exit(1)
    
    elif args.command == "verify":
        from src.verification import verify_speaker
        try:
            result = verify_speaker(args.audio, args.user, threshold=args.threshold)
            status = "✓ VERIFIED" if result.is_verified else "✗ DENIED"
            print(f"{status}")
            print(f"  User: {result.user_id}")
            print(f"  Score: {result.score:.4f}")
            print(f"  Threshold: {result.threshold:.4f}")
        except ValueError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    
    elif args.command == "identify":
        from src.verification import identify_speaker
        matches = identify_speaker(args.audio, threshold=args.threshold, top_n=args.top)
        if matches:
            print(f"Top {len(matches)} matches:")
            for i, match in enumerate(matches, 1):
                print(f"  {i}. {match['name']} ({match['user_id']}) - Score: {match['score']:.4f}")
        else:
            print("No matches found above threshold")
    
    elif args.command == "list":
        from src.enrollment import list_enrolled_users
        users = list_enrolled_users()
        if users:
            print(f"Enrolled users ({len(users)}):")
            for u in users:
                print(f"  • {u['name']} ({u['id']}) - {u['num_samples']} samples")
        else:
            print("No users enrolled")
    
    elif args.command == "delete":
        from src.enrollment import delete_enrollment
        if delete_enrollment(args.user):
            print(f"✓ Deleted user '{args.user}'")
        else:
            print(f"✗ User '{args.user}' not found")
            sys.exit(1)
    
    elif args.command == "serve":
        import uvicorn
        print(f"Starting API server at http://{args.host}:{args.port}")
        print(f"Swagger UI available at http://{args.host}:{args.port}/docs")
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )


if __name__ == "__main__":
    main()
